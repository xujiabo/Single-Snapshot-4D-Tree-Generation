"""KPConv point-cloud conditioning encoder."""
import torch
from torch import nn

from .kp_conv import KPFCN


def farthest_point_indices(points, count):
    """Deterministic FPS; start at the point farthest from the cloud centroid."""
    if len(points) <= count:
        return torch.arange(len(points), device=points.device)
    with torch.no_grad():
        distance = torch.full((len(points),), float("inf"), device=points.device)
        farthest = (points - points.mean(0)).square().sum(-1).argmax()
        indices = torch.empty(count, dtype=torch.long, device=points.device)
        for i in range(count):
            indices[i] = farthest
            distance = torch.minimum(distance, (points - points[farthest]).square().sum(-1))
            # Prevent duplicate selection, including coincident input points.
            distance[farthest] = -1
            farthest = distance.argmax()
    return indices


class KPConvTransformer(nn.Module):
    """Return feats (B,P,D), positions (B,P,3), and valid mask (B,P).

    At most n_target=256 superpoints are retained per cloud. Shorter clouds are
    padded only to the batch maximum; True in mask denotes a real superpoint.
    FPS is deterministic in both training and evaluation.
    """
    def __init__(self, d_model=384, n_heads=6, n_layers=4, n_target=256,
                 dim_feedforward=None, dropout=0.0, max_sp_per_batch=None):
        super().__init__()
        if n_target < 1:
            raise ValueError("n_target must be positive.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.kpconv = KPFCN()
        self.d_model = d_model
        self.n_target = int(n_target)
        if max_sp_per_batch is not None:
            self.n_target = min(self.n_target, int(max_sp_per_batch))
            if self.n_target < 1:
                raise ValueError("max_sp_per_batch must be positive.")
        self.feat_proj = nn.Identity() if d_model == 384 else nn.Linear(384, d_model)
        self.pos_embed = nn.Sequential(nn.Linear(3, d_model // 2), nn.GELU(),
                                       nn.Linear(d_model // 2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)

    def _pad_and_stack(self, feat_list, sp_list):
        batch_size = len(feat_list)
        max_points = min(max(len(feature) for feature in feat_list), self.n_target)
        features = feat_list[0].new_zeros(batch_size, max_points, feat_list[0].shape[-1])
        positions = sp_list[0].new_zeros(batch_size, max_points, 3)
        mask = torch.zeros(batch_size, max_points, dtype=torch.bool, device=features.device)
        for i, (feature, point) in enumerate(zip(feat_list, sp_list)):
            index = farthest_point_indices(point, max_points)
            n = len(index)
            features[i, :n], positions[i, :n] = feature[index], point[index]
            mask[i, :n] = True
        return features, positions, mask

    def forward(self, pcd_list):
        feat_list, sp_list = self.kpconv(pcd_list)
        features, positions, mask = self._pad_and_stack(feat_list, sp_list)
        tokens = self.feat_proj(features) + self.pos_embed(positions)
        encoded = self.transformer(tokens, src_key_padding_mask=~mask)
        return encoded.masked_fill(~mask.unsqueeze(-1), 0), positions, mask
