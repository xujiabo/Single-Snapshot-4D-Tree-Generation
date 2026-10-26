"""One backbone for coordinate flow matching and endpoint parent/birth decoding.

The ELF idea used here is shared-weight, endpoint-only discretization. As in
the 4DTree paper, the continuous state is XYZ (not a language embedding), the
network predicts velocity, and there is no self-conditioning or token rounding.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import KPConvTransformer


class Block(nn.Module):
    """AdaLN-Zero block: bidirectional nodes, read-only point-cloud memory."""
    def __init__(self, d, nhead, mlp=4):
        super().__init__()
        self.nhead, self.hd = nhead, d // nhead
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def _attn(self, x, mask):
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.nhead, self.hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(batch, length, dim))

    def forward(self, x, c, mask=None):
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, -1)
        x = x + g1 * self._attn(self.n1(x) * (1 + s1) + b1, mask)
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


def timestep_emb(t, d):
    half = d // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    phase = t.float()[:, None] * freq[None] * 1000.0
    out = torch.cat([phase.sin(), phase.cos()], -1)
    return F.pad(out, (0, d - out.shape[-1]))


class TreeFlow(nn.Module):
    """Unified 4DTree model. Birth labels and decoded births are 1-based (1..35).

    All generated outputs retain a batch dimension, including single-tree samples.
    Species -1/None uses a learned null label; known labels are 0..n_species-1.
    """
    def __init__(self, d=384, nhead=6, layers=12, n_species=5, n_nodes=1024,
                 pcd_token_num=256, n_birth=35, pc_layers=4):
        super().__init__()
        if d < 2 or nhead < 1 or d % nhead or min(layers, n_nodes, n_birth, pc_layers) < 1:
            raise ValueError("Invalid model dimensions: d must be divisible by nhead.")
        self.d, self.N, self.P = d, n_nodes, pcd_token_num
        self.n_species, self.n_birth = n_species, n_birth
        self.coord_in = nn.Linear(3, d)
        self.slot_pos = nn.Parameter(torch.randn(1, n_nodes, d) * 0.02)
        self.type_emb = nn.Embedding(2, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.species_emb = nn.Embedding(n_species + 1, d)  # last entry = null
        self.pcd_encoder = KPConvTransformer(d_model=d, n_heads=nhead, n_layers=pc_layers,
                                             dim_feedforward=4 * d, n_target=pcd_token_num)
        self.pc_pos = nn.Sequential(nn.Linear(3, d // 2), nn.GELU(), nn.Linear(d // 2, d))
        self.pc_null = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nhead) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.vel_head = nn.Linear(d, 3)
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.birth_head = nn.Linear(d, n_birth)
        nn.init.zeros_(self.vel_head.weight)
        nn.init.zeros_(self.vel_head.bias)
        causal = torch.tril(torch.ones(n_nodes, n_nodes, dtype=torch.bool), diagonal=-1)
        causal[0, 0] = True  # finite dummy root row; it is excluded from parent CE
        self.register_buffer("parent_mask", causal, persistent=False)

    def _species(self, species, batch, device, drop=None):
        if species is None:
            sp = torch.full((batch,), self.n_species, dtype=torch.long, device=device)
        else:
            sp = torch.as_tensor(species, dtype=torch.long, device=device).reshape(-1)
            if sp.numel() == 1:
                sp = sp.expand(batch)
            if sp.numel() != batch or ((sp < -1) | (sp >= self.n_species)).any():
                raise ValueError("species must contain one label per tree (or a scalar), -1 or 0..4.")
            sp = torch.where(sp == -1, self.n_species, sp)
        if drop is not None:
            sp = torch.where(drop, self.n_species, sp)
        return sp

    def encode_pc(self, pcd_list, batch_size=None, device=None, drop=None):
        """Return (tokens, valid_mask); cache this tuple across ODE steps.

        A dropped cloud has exactly one visible null token. Keeping the original
        valid-token count here would expose cloud size to the CFG null branch.
        """
        device = self.slot_pos.device if device is None else device
        batch = batch_size if batch_size is not None else (len(pcd_list) if pcd_list is not None else 1)
        if pcd_list is None:
            pc = self.pc_null.expand(batch, 1, -1)
            mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
        else:
            if len(pcd_list) != batch:
                raise ValueError("Point-cloud batch size does not match coordinate batch size.")
            feats, pos, mask = self.pcd_encoder(pcd_list)
            pc = feats + self.pc_pos(pos)
        if drop is not None:
            pc, mask = self._drop_pc((pc, mask), drop, typed=False)
        return pc + self.type_emb.weight[0], mask

    def _drop_pc(self, pc_cache, drop, typed=True):
        pc, mask = pc_cache
        null = self.pc_null + self.type_emb.weight[0] if typed else self.pc_null
        pc = torch.where(drop[:, None, None], null.expand_as(pc), pc)
        null_mask = torch.zeros_like(mask)
        null_mask[:, 0] = True
        return pc, torch.where(drop[:, None], null_mask, mask)

    def cond_mask(self, pc_mask):
        batch, num_pc = pc_mask.shape
        mask = torch.zeros(batch, num_pc + self.N, num_pc + self.N,
                           dtype=torch.bool, device=pc_mask.device)
        mask[:, :, :num_pc] = pc_mask[:, None, :]
        mask[:, num_pc:, num_pc:] = True
        return mask[:, None]

    def encode(self, x_t, t, species=None, pcd_list=None, drop_pc=None,
               drop_species=None, pc_cache=None):
        """The single backbone shared by flow training and endpoint decoding."""
        batch = x_t.shape[0]
        if x_t.ndim != 3 or x_t.shape[1:] != (self.N, 3):
            raise ValueError("Coordinates must have shape (B, n_nodes, 3).")
        t = torch.as_tensor(t, device=x_t.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError("Time must be a scalar or have shape (B,).")
        sp = self._species(species, batch, x_t.device, drop_species)
        c = (self.time_mlp(timestep_emb(t, self.d).to(x_t.dtype)) + self.species_emb(sp))[:, None]
        if pc_cache is None:
            pc, pc_mask = self.encode_pc(pcd_list, batch, x_t.device, drop_pc)
        else:
            pc, pc_mask = pc_cache if drop_pc is None else self._drop_pc(pc_cache, drop_pc)
        nodes = self.coord_in(x_t) + self.slot_pos + self.type_emb.weight[1]
        h = torch.cat([pc, nodes], dim=1)
        mask = self.cond_mask(pc_mask)
        for block in self.blocks:
            h = block(h, c, mask)
        return self.norm(h[:, pc.shape[1]:])

    def forward(self, x_t, t, species=None, pcd_list=None, drop_pc=None,
                drop_species=None, pc_cache=None):
        h = self.encode(x_t, t, species, pcd_list, drop_pc, drop_species, pc_cache)
        return self.vel_head(h)

    def pointer_logits(self, h, coords):
        q, k = self.q_proj(h).float(), self.k_proj(h).float()
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d)
        d2 = torch.cdist(coords.float(), coords.float()).square()
        score = score - F.softplus(self.log_alpha.float()) * d2
        return score.masked_fill(~self.parent_mask[None], float("-inf"))

    def loss(self, species, coords, parents, birth, pcd_list, tau=0.98,
             sigma_max=0.04, lambda_birth=1.0, p_drop_pc=0.1,
             p_drop_species=0.1, regime="mixed"):
        """Eq. (7), with independently selected per-example training regimes.

        mixed: u~U(0,1), flow for u<tau, endpoint for u>=tau (at time 1).
        flow/terminal force a branch for diagnostics. loss averages over the full
        batch; flow/parent/birth metrics average only their participating samples.
        No forced terminal example changes the paper's 98%/2% regime probability.
        """
        if not 0 < tau < 1 or sigma_max < 0 or lambda_birth < 0:
            raise ValueError("Require 0<tau<1, sigma_max>=0 and lambda_birth>=0.")
        if not 0 <= p_drop_pc <= 1 or not 0 <= p_drop_species <= 1:
            raise ValueError("Drop probabilities must lie in [0,1].")
        batch, num_nodes, _ = coords.shape
        if batch < 1 or num_nodes != self.N or parents.shape != coords.shape[:2] or birth.shape != parents.shape:
            raise ValueError("coords, parents and birth must have matching nonempty (B,N) dimensions.")
        if ((birth < 1) | (birth > self.n_birth)).any():
            raise ValueError("Birth targets must be 1-based integers in 1..n_birth.")
        ids = torch.arange(self.N, device=coords.device)[None]
        if ((parents[:, 1:] < 0) | (parents[:, 1:] >= ids[:, 1:])).any():
            raise ValueError("Every non-root parent must precede its child in DFS order.")
        u = torch.rand(batch, device=coords.device)
        if regime == "mixed":
            terminal = u >= tau
        elif regime == "flow":
            u = u * tau
            terminal = torch.zeros(batch, dtype=torch.bool, device=coords.device)
        elif regime == "terminal":
            terminal = torch.ones(batch, dtype=torch.bool, device=coords.device)
        else:
            raise ValueError("regime must be 'mixed', 'flow' or 'terminal'.")
        noise = torch.randn_like(coords)
        x_flow = (1 - u[:, None, None]) * noise + u[:, None, None] * coords
        sigma = torch.rand(batch, 1, 1, device=coords.device) * sigma_max
        x_terminal = coords + sigma * torch.randn_like(coords)
        x = torch.where(terminal[:, None, None], x_terminal, x_flow)
        t = torch.where(terminal, torch.ones_like(u), u)
        drop_pc = torch.rand(batch, device=coords.device) < p_drop_pc
        drop_sp = torch.rand(batch, device=coords.device) < p_drop_species
        h = self.encode(x, t, species, pcd_list, drop_pc, drop_sp)
        zero = h.sum() * 0.0
        flow_loss = parent_loss = birth_loss = zero
        parent_acc = parent_acc_hard = birth_acc = birth_mae = zero.detach()
        total = zero
        flow = ~terminal
        if flow.any():
            per_flow = (self.vel_head(h[flow]).float() - (coords - noise)[flow].float()).square().mean((1, 2))
            flow_loss = per_flow.mean()
            total = total + per_flow.sum()
        if terminal.any():
            th, tx = h[terminal], x[terminal]
            target_parent, target_birth = parents[terminal].long(), birth[terminal].long() - 1
            logits = self.pointer_logits(th, tx)
            birth_logits = self.birth_head(th).float()
            per_birth = F.cross_entropy(birth_logits.transpose(1, 2), target_birth, reduction="none").mean(1)
            if self.N > 1:
                per_parent = F.cross_entropy(logits[:, 1:].transpose(1, 2), target_parent[:, 1:], reduction="none").mean(1)
                correct = logits[:, 1:].argmax(-1) == target_parent[:, 1:]
                parent_acc = correct.float().mean()
                hard = target_parent[:, 1:] != ids[:, 1:] - 1
                if hard.any():
                    parent_acc_hard = correct[hard].float().mean()
            else:
                per_parent = torch.zeros_like(per_birth)
            parent_loss, birth_loss = per_parent.mean(), per_birth.mean()
            pred_birth = birth_logits.argmax(-1)
            birth_acc = (pred_birth == target_birth).float().mean()
            birth_mae = (pred_birth - target_birth).float().abs().mean()
            total = total + (per_parent + lambda_birth * per_birth).sum()
        return {"loss": total / batch, "flow": flow_loss.detach(),
                "parent": parent_loss.detach(), "birth": birth_loss.detach(),
                "parent_acc": parent_acc.detach(), "parent_acc_hard": parent_acc_hard.detach(),
                "birth_acc": birth_acc.detach(), "birth_mae": birth_mae.detach(),
                "flow_count": flow.sum(), "terminal_count": terminal.sum()}

    @torch.no_grad()
    def decode(self, coords, species=None, pcd_list=None, pc_cache=None):
        """Decode once at t=1, then enforce parent-before-child birth monotonicity."""
        h = self.encode(coords, 1.0, species, pcd_list, pc_cache=pc_cache)
        parents = self.pointer_logits(h, coords).argmax(-1)
        parents[:, 0] = -1
        birth_raw = self.birth_head(h).argmax(-1) + 1
        birth = birth_raw.clone()
        for i in range(1, self.N):
            parent_birth = birth.gather(1, parents[:, i:i + 1]).squeeze(1)
            birth[:, i] = torch.maximum(birth[:, i], parent_birth)
        return {"coords": coords, "parents": parents, "birth": birth, "birth_raw": birth_raw}

    @torch.no_grad()
    def sample(self, species=None, pcd_list=None, steps=25, guidance=2.0, seed=None):
        """Euler noise->XYZ then one endpoint decode; output tensors are (B,N,...).

        guidance is w in Eq. (8): 0 means pure conditional;
        2 means v_null + 3*(v_cond-v_null). No coordinate clipping is applied.
        """
        if not isinstance(steps, int) or steps < 1 or not math.isfinite(guidance) or guidance < 0:
            raise ValueError("steps must be a positive integer and guidance must be finite and >=0.")
        device, dtype = self.slot_pos.device, self.slot_pos.dtype
        if isinstance(pcd_list, torch.Tensor):
            pcd_list = [pcd_list] if pcd_list.ndim == 2 else list(pcd_list)
        batch = len(pcd_list) if pcd_list is not None else (torch.as_tensor(species).numel() if species is not None else 1)
        if batch < 1:
            raise ValueError("Cannot sample an empty batch.")
        self._species(species, batch, device)  # validate before expensive PC encoding
        was_training = self.training
        self.eval()
        try:
            pc_cache = self.encode_pc(pcd_list, batch, device)
            drop = torch.ones(batch, dtype=torch.bool, device=device)
            null_cache = self._drop_pc(pc_cache, drop)
            generator = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None
            x = torch.randn(batch, self.N, 3, device=device, dtype=dtype, generator=generator)
            dt = 1.0 / steps
            for k in range(steps):
                t = torch.full((batch,), k * dt, device=device)
                v_cond = self(x, t, species, pc_cache=pc_cache)
                if guidance:
                    v_null = self(x, t, None, pc_cache=null_cache)
                    velocity = v_null + (1.0 + guidance) * (v_cond - v_null)
                else:
                    velocity = v_cond
                x = x + dt * velocity
            return self.decode(x, species, pc_cache=pc_cache)
        finally:
            self.train(was_training)
