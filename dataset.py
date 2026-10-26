"""4DTree data: final skeleton + real birth labels + a scan at an unknown stage.

Run this file directly to inspect one synthetic tree example. The first visit to
a tree/stage/scanner builds a real ray-cast LiDAR scan; later visits use the cache.
All geometry is transformed using ONLY the observed point cloud's bounding box.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from tree.tree_utils import snapshot_at_time
from tree.species_presets import SPECIES_PRESETS

PROJECT_DIR = Path(__file__).resolve().parent
SPECIES_NAMES = ("oak", "birch", "maple", "pine", "willow")
SPECIES = {name: i for i, name in enumerate(SPECIES_NAMES)}
N_NODES = 1024
N_BIRTH = 35
SCAN_CONFIGS = (("tls", "sparse"), ("als", "dense"), ("als", "medium"),
                ("mls", "sparse"), ("mls", "medium"))
SCAN_VERSION = "lidar-leaf-seeded-1"


def _points(xyz):
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Point cloud must be a nonempty (N, 3) array.")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinity.")
    return points


def normalize_point_cloud(pc):
    """Isotropic scan-only fit, all points in [-1, 1] and maximum z exactly 1.

    x/y are centered in their observed bounding box. z is anchored at the scan
    top, so the max-height matching rule also works for crowns wider than tall.
    ``scale`` is the original-world half extent, not its reciprocal.
    """
    pc = _points(pc)
    lo, hi = pc.min(0).astype(np.float64), pc.max(0).astype(np.float64)
    scale = float((hi - lo).max() * 0.5)
    if scale < 1e-8:
        raise ValueError("Cannot normalize a point cloud with zero spatial extent.")
    center = np.array([(lo[0] + hi[0]) * .5, (lo[1] + hi[1]) * .5,
                       hi[2] - scale], np.float64)
    meta = {"center": center, "scale": scale}
    return normalize_pc_with_meta(pc, meta), meta


def normalize_pc_with_meta(xyz, meta):
    """Apply the observed scan's transform to any xyz array; never clip targets."""
    points = np.asarray(xyz, np.float64)
    if "rotation" in meta:
        points = points @ np.asarray(meta["rotation"]).T
    return ((points - np.asarray(meta["center"])) /
            float(meta["scale"])).astype(np.float32)


def renormalize_pc_with_meta(xyz, meta):
    """Return coordinates to the original input frame, including augmentation."""
    points = np.asarray(xyz, np.float64) * float(meta["scale"]) + meta["center"]
    if "rotation" in meta:
        points = points @ np.asarray(meta["rotation"])
    return points.astype(np.float32)


def npy_to_nodes(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 9 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Tree file must contain a finite nonempty (N, 9) array.")
    parents = a[:, 8].astype(np.int64)
    if not np.equal(a[:, 8], parents).all() or np.count_nonzero(parents == -1) != 1:
        raise ValueError("Tree must have integer parent indices and exactly one root.")
    if parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents >= np.arange(len(a))):
        raise ValueError("Expected parent-before-child tree with root at index 0.")
    birth = a[:, 7].astype(np.int64)
    if not np.equal(a[:, 7], birth).all():
        raise ValueError("Birth steps must be integers, not normalized fractions.")
    for i, parent in enumerate(parents):
        if birth[i] <= 0:
            birth[i] = birth[parent] if parent >= 0 else 1
        if parent >= 0 and birth[i] < birth[parent]:
            raise ValueError("Ground-truth birth step is earlier than its parent's.")
    return [{"start_xyz": row[:3].astype(np.float32).copy(),
             "xyz": row[3:6].astype(np.float32).copy(), "radius": float(row[6]),
             "birth_step": int(birth[i]), "parent_idx": int(parents[i])}
            for i, row in enumerate(a)]


def _dfs_order(nodes):
    """Canonical order is fixed before random rotation."""
    children = [[] for _ in nodes]
    for i, nd in enumerate(nodes):
        if nd["parent_idx"] >= 0:
            children[nd["parent_idx"]].append(i)
    order, stack = [], [0]
    while stack:
        parent = stack.pop()
        order.append(parent)
        def key(child):
            delta = nodes[child]["xyz"] - nodes[parent]["xyz"]
            return -nodes[child]["radius"], float(np.arctan2(delta[1], delta[0])), child
        stack.extend(reversed(sorted(children[parent], key=key)))
    if len(order) != len(nodes):
        raise ValueError("Disconnected or cyclic tree.")
    return order


def _remap_nodes(nodes, keep):
    keep = sorted(keep)
    lookup = {old: new for new, old in enumerate(keep)}
    result = []
    for old in keep:
        node = dict(nodes[old])
        parent = node["parent_idx"]
        while parent >= 0 and parent not in lookup:
            parent = nodes[parent]["parent_idx"]
        node["parent_idx"] = lookup.get(parent, -1)
        if parent >= 0:
            node["start_xyz"] = nodes[parent]["xyz"].copy()
        result.append(node)
    return result


def canonicalize_nodes(nodes, target=1024, seed=42):
    """Topology-preserving resampling with a local RNG and genuine short-tree edges.

    The normal 1024-node data retains all roots/leaves/branch points. For smaller
    debug targets where these already exceed the budget, prioritize thick nodes
    while always retaining the root and tallest endpoint. Short trees are expanded
    by splitting long edges; inserted points inherit that edge's birth step.
    """
    if target < 2:
        raise ValueError("n_nodes must be at least 2.")
    nodes = [dict(nd, xyz=np.array(nd["xyz"], np.float32).copy(),
                  start_xyz=np.array(nd["start_xyz"], np.float32).copy()) for nd in nodes]
    rng = np.random.default_rng(seed)
    if len(nodes) > target:
        child_count = np.bincount([nd["parent_idx"] for nd in nodes if nd["parent_idx"] >= 0],
                                 minlength=len(nodes))
        key = np.flatnonzero(child_count != 1).tolist()
        mandatory = {0, int(np.argmax([nd["xyz"][2] for nd in nodes]))}
        key = sorted(set(key) | mandatory)
        if len(key) > target:
            ranked = sorted(set(key) - mandatory, key=lambda i: (-nodes[i]["radius"], i))
            keep = mandatory | set(ranked[:target - len(mandatory)])
        else:
            candidates = np.array(sorted(set(range(len(nodes))) - set(key)))
            remain = target - len(key)
            radii = np.array([max(nodes[i]["radius"], 1e-12) for i in candidates])
            selected = rng.choice(candidates, remain, replace=False, p=radii / radii.sum())
            keep = set(key) | set(selected.tolist())
        nodes = _remap_nodes(nodes, keep)
    while len(nodes) < target:
        lengths = [float(np.linalg.norm(nd["xyz"] - nd["start_xyz"])) for nd in nodes]
        child = int(np.argmax(lengths))
        if lengths[child] <= 1e-10:
            raise ValueError("Cannot expand a tree consisting only of zero-length edges.")
        nd = nodes[child]
        midpoint = .5 * (nd["start_xyz"] + nd["xyz"])
        inserted = dict(nd, xyz=midpoint.copy(), start_xyz=nd["start_xyz"].copy())
        nd["parent_idx"] = len(nodes)
        nd["start_xyz"] = midpoint.copy()
        nodes.append(inserted)
    roots = [i for i, nd in enumerate(nodes) if nd["parent_idx"] == -1]
    # Splitting the root edge can move the root to the appended slot.
    if roots[0] != 0:
        first = roots[0]
        order = [first] + [i for i in range(len(nodes)) if i != first]
        lookup = {old: new for new, old in enumerate(order)}
        nodes = [dict(nodes[i], parent_idx=lookup.get(nodes[i]["parent_idx"], -1)) for i in order]
    seq = _dfs_order(nodes)
    lookup = {old: new for new, old in enumerate(seq)}
    return [dict(nodes[i], parent_idx=lookup.get(nodes[i]["parent_idx"], -1)) for i in seq]


def _stable_seed(text):
    return int.from_bytes(hashlib.sha256(str(text).encode("utf-8")).digest()[:4], "little")


def simulate_scan(nodes, species, step, mode="tls", density="sparse", seed=42):
    """Build bark and leaf meshes and ray-cast the selected growth snapshot."""
    from tree.tree3D_utils import build_tree_mesh, build_leaves_mesh
    from tree.lidar_utils_v2 import simulate
    snapshot = snapshot_at_time(nodes, step, pipe_exponent=SPECIES_PRESETS[species]["PIPE_EXPONENT"])
    if not snapshot:
        raise ValueError(f"No nodes alive at growth step {step}.")
    mesh = build_tree_mesh(snapshot, radial_segments=8, radius_scale=2.0)
    mesh += build_leaves_mesh(snapshot, species=species, seed=seed)
    pc = np.asarray(simulate(mesh, mode, density, seed=seed).points, np.float32)
    if len(pc) < 8:
        raise RuntimeError(f"LiDAR scan has only {len(pc)} returns ({species}, step={step}, "
                           f"{mode}/{density}); choose a denser scan preset.")
    return _points(pc)


def _cap_points(pc, max_points, rng):
    if len(pc) <= max_points:
        return pc.copy()
    # Keep all extrema: subsampling must not quietly change the normalizing box.
    extremes = np.unique(np.r_[pc.argmin(0), pc.argmax(0)])
    available = np.setdiff1d(np.arange(len(pc)), extremes, assume_unique=True)
    chosen = rng.choice(available, max_points - len(extremes), replace=False)
    return pc[np.r_[extremes, chosen]].copy()


def prepare_point_cloud(pc, max_points=4096, voxel_size=.025, seed=42):
    """Shared training/inference preprocessing; preserve original scan extrema.

    Fit the scan-only transform first, then voxel-filter in normalized units and
    cap the count. Extrema are explicitly retained, so the delivered point cloud
    still spans the same bounding box and its highest z remains exactly 1.
    Rotation augmentation, when used, must be applied before calling this.
    """
    if max_points < 8 or voxel_size < 0:
        raise ValueError("Require max_points>=8 and voxel_size>=0.")
    points, meta = normalize_point_cloud(pc)
    if voxel_size > 0:
        cells = np.floor(points / voxel_size).astype(np.int64)
        _, selected = np.unique(cells, axis=0, return_index=True)
        extrema = np.r_[points.argmin(0), points.argmax(0)]
        points = points[np.unique(np.r_[selected, extrema])]
    return _cap_points(points, max_points, np.random.default_rng(seed)), meta


class TreeDataset(Dataset):
    def __init__(self, root=PROJECT_DIR / "L-system-generated", n_nodes=1024, n_birth=35,
                 augment=True, seed=42, max_points=4096, cache_dir=None,
                 scan_fractions=None, scan_configs=SCAN_CONFIGS, split="all", val_fraction=.1,
                 species_list=SPECIES_NAMES, n_per=None, max_trees=None, voxel_size=.025):
        self.root = Path(root).resolve()
        self.n_nodes, self.n_birth = int(n_nodes), int(n_birth)
        self.augment, self.seed, self.max_points = bool(augment), int(seed), int(max_points)
        self.voxel_size = float(voxel_size)
        if self.voxel_size < 0:
            raise ValueError("voxel_size must be nonnegative.")
        if self.n_nodes < 2 or self.n_birth < 1 or self.max_points < 8:
            raise ValueError("Require n_nodes>=2, n_birth>=1, max_points>=8.")
        if split not in ("all", "train", "val") or not 0 <= val_fraction < 1:
            raise ValueError("split must be all/train/val, and val_fraction in [0,1).")
        self.scan_fractions = None if scan_fractions is None else tuple(scan_fractions)
        if self.scan_fractions is not None and (not self.scan_fractions or
                any(not 0 < fraction <= 1 for fraction in self.scan_fractions)):
            raise ValueError("scan_fractions must be nonempty fractions in (0,1].")
        self.scan_configs = tuple((mode.lower(), density.lower()) for mode, density in scan_configs)
        if not self.scan_configs or any(mode not in ("tls", "als", "mls") or
                density not in ("sparse", "medium", "dense") for mode, density in self.scan_configs):
            raise ValueError("Unknown LiDAR mode or density.")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root.parent / "dataset_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.files = []
        for species in species_list:
            if species not in SPECIES:
                raise ValueError(f"Unknown species {species}.")
            files = sorted((self.root / species).glob("*.npy"),
                           key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem))
            if n_per is not None:
                files = files[:int(n_per)]
            order = np.random.default_rng(_stable_seed(f"split-{seed}-{species}")).permutation(len(files))
            n_val = min(len(files) - 1, max(1, round(len(files) * val_fraction))) if len(files) > 1 and val_fraction else 0
            val_ids = set(order[:n_val].tolist())
            files = [path for i, path in enumerate(files)
                     if split == "all" or ((i in val_ids) == (split == "val"))]
            self.files.extend((path, species) for path in files)
        if max_trees is not None:
            self.files = self.files[:int(max_trees)]
        if not self.files:
            raise ValueError(f"No trees for split={split} in {self.root}. Validation needs at least two trees per species.")
        self._tree_cache = OrderedDict()

    def __len__(self):
        return len(self.files)

    def _load_tree(self, index):
        if index not in self._tree_cache:
            path, species = self.files[index]
            source_bytes = path.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            nodes = npy_to_nodes(np.load(path, allow_pickle=False))
            max_birth = max(nd["birth_step"] for nd in nodes)
            if max_birth > self.n_birth:
                raise ValueError(f"{path}: birth={max_birth} exceeds n_birth={self.n_birth}.")
            canonical = canonicalize_nodes(nodes, self.n_nodes, _stable_seed(f"{self.seed}-{source_hash}"))
            self._tree_cache[index] = nodes, canonical, source_hash, max_birth
            if len(self._tree_cache) > 64:
                self._tree_cache.popitem(last=False)
        self._tree_cache.move_to_end(index)
        return self._tree_cache[index]

    def __getitem__(self, index):
        path, species = self.files[index]
        nodes, canonical, source_hash, max_birth = self._load_tree(index)
        # NumPy's per-worker seed is initialized by PyTorch DataLoader. Validation
        # uses only a tree-specific seed and is independent of global RNG state.
        sample_seed = int(np.random.randint(0, 2 ** 32 - 1, dtype=np.uint32)) if self.augment else _stable_seed(f"{self.seed}-{source_hash}-eval")
        rng = np.random.default_rng(sample_seed)
        fraction = float(rng.uniform(.5, 1.)) if self.scan_fractions is None else float(rng.choice(self.scan_fractions))
        # Quantize upward so uniform [.5, 1] also gives the mature snapshot a
        # nonzero probability. floor would never reach max_birth for random g.
        step = max(1, min(max_birth, math.ceil(fraction * max_birth)))
        mode, density = self.scan_configs[int(rng.integers(len(self.scan_configs)))]
        scan_config = dict(version=SCAN_VERSION, source=source_hash, species=species,
                           step=step, mode=mode, density=density, seed=self.seed)
        digest = hashlib.sha256(json.dumps(scan_config, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = self.cache_dir / f"{species}-{path.stem}-t{step}-{mode}-{density}-{digest}.npy"
        if cache_path.exists():
            pc = _points(np.load(cache_path, allow_pickle=False))
        else:
            pc = simulate_scan(nodes, species, step, mode, density, seed=_stable_seed(digest))
            # Atomic replacement allows independent DataLoader workers to build a
            # shared cache safely; a partial .npy never becomes visible.
            temp = cache_path.with_suffix(f".{os.getpid()}.{sample_seed}.tmp")
            with temp.open("wb") as handle:
                np.save(handle, pc, allow_pickle=False)
            os.replace(temp, cache_path)
        coords = np.array([nd["xyz"] for nd in canonical], np.float32)
        parents = np.array([nd["parent_idx"] for nd in canonical], np.int64)
        birth = np.array([nd["birth_step"] for nd in canonical], np.int64)
        rotation = np.eye(3, dtype=np.float32)
        if self.augment:
            theta = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            pc, coords = pc @ rotation.T, coords @ rotation.T
        pc, meta = prepare_point_cloud(pc, self.max_points, self.voxel_size, sample_seed)
        coords = normalize_pc_with_meta(coords, meta)
        meta.update(rotation=rotation, source=str(path), species=species,
                    scan_step=step, scan_fraction=step / max_birth,
                    requested_fraction=fraction, max_birth=max_birth,
                    scan_mode=mode, scan_density=density, cache_path=str(cache_path))
        return dict(species=SPECIES[species], coords=coords, parents=parents,
                    birth=birth, pcd=pc, meta=meta)


def collate(batch):
    return dict(species=torch.tensor([sample["species"] for sample in batch], dtype=torch.long),
                coords=torch.from_numpy(np.stack([sample["coords"] for sample in batch])),
                parents=torch.from_numpy(np.stack([sample["parents"] for sample in batch])),
                birth=torch.from_numpy(np.stack([sample["birth"] for sample in batch])),
                pcd_list=[torch.from_numpy(sample["pcd"]) for sample in batch],
                metas=[sample["meta"] for sample in batch])


if __name__ == "__main__":
    import time
    started = time.perf_counter()
    data = TreeDataset(augment=False, scan_fractions=(.7,), scan_configs=(("tls", "sparse"),))
    sample = data[0]
    print(f"Trees: {len(data)}; first sample: {time.perf_counter() - started:.2f}s")
    print("coords:", sample["coords"].shape, "point cloud:", sample["pcd"].shape)
    print("birth:", sample["birth"].min(), "..", sample["birth"].max())
    print("scan bounds:", sample["pcd"].min(0), sample["pcd"].max(0))
    print("final-tree bounds (unclipped):", sample["coords"].min(0), sample["coords"].max(0))
    print("metadata:", sample["meta"])
