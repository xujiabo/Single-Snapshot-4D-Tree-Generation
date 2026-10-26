"""KPConv encoder with portable, device-independent geometry preprocessing.

Locally compiled C++ extensions are used when available. Otherwise,
NumPy voxel barycentres and scipy.spatial.cKDTree radius neighbours perform the
same preprocessing; the learned KPConv network is identical in both cases.
"""
import warnings

import numpy as np
import torch
from torch import nn

from .kpconv.blocks import block_decider

try:
    from .kpconv.cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
    from .kpconv.cpp_wrappers.cpp_neighbors import radius_neighbors as cpp_neighbors
    PREPROCESS_BACKEND = "native C++"
except (ImportError, OSError):
    cpp_subsampling = cpp_neighbors = None
    PREPROCESS_BACKEND = "NumPy voxel + SciPy cKDTree"
    warnings.warn(
        "KPConv native extensions are unavailable for this Python/platform. "
        "Using NumPy voxel barycentres and SciPy cKDTree radius neighbours; "
        "the KPConv model is unchanged.", RuntimeWarning
    )


class Config:
    def __init__(self):
        self.num_layers = 4
        self.in_points_dim = 3
        self.first_feats_dim = 128
        self.first_subsampling_dl = 0.025
        self.in_feats_dim = 1
        self.final_feats_dim = 384
        self.conv_radius = 2.5
        self.deform_radius = 5.0
        self.num_kernel_points = 15
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True  # The blocks use GroupNorm.
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False
        self.neighborhood_limits = [64, 64, 64, 64]
        self.architecture = [
            "simple", "resnetb", "resnetb", "resnetb_strided",
            "resnetb", "resnetb", "resnetb_strided",
            "resnetb", "resnetb", "resnetb_strided",
            "resnetb", "resnetb", "last_unary",
        ]


def _numpy(value, dtype):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _voxel_subsample(points, lengths, features, labels, sample_dl, max_points):
    """Batch-separated voxels: average XYZ/features, majority-vote labels.

    C++ uses an unordered map; this implementation emits lexicographic voxel
    order. Barycentres and neighbourhood semantics are the same, although
    floating-point summation order and tied-label selection can differ.
    """
    result_points, result_lengths, result_features, result_labels = [], [], [], []
    offset = 0
    for length in lengths:
        length = int(length)
        cloud = points[offset:offset + length]
        if length == 0:
            raise ValueError("Each point cloud must contain at least one point.")
        origin = np.floor(cloud.min(axis=0) / sample_dl) * sample_dl
        voxels = np.floor((cloud - origin) / sample_dl).astype(np.int64)
        _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
        count = len(counts)
        limit = min(count, max_points) if max_points > 0 else count
        sums = np.zeros((count, 3), dtype=np.float32)
        np.add.at(sums, inverse, cloud)
        result_points.append((sums / counts[:, None]).astype(np.float32)[:limit])
        result_lengths.append(limit)
        if features is not None:
            feats = features[offset:offset + length]
            sums = np.zeros((count,) + feats.shape[1:], dtype=np.float32)
            np.add.at(sums, inverse, feats)
            divisor = counts.reshape((-1,) + (1,) * (feats.ndim - 1))
            result_features.append((sums / divisor).astype(np.float32)[:limit])
        if labels is not None:
            lab = labels[offset:offset + length]
            flat = lab.reshape(length, -1)
            voted = np.empty((limit, flat.shape[1]), dtype=np.int32)
            for cell in range(limit):
                for dim in range(flat.shape[1]):
                    values, freq = np.unique(flat[inverse == cell, dim], return_counts=True)
                    voted[cell, dim] = values[freq.argmax()]
            result_labels.append(voted.reshape((limit,) + lab.shape[1:]))
        offset += length
    result = [np.concatenate(result_points), np.asarray(result_lengths, dtype=np.int32)]
    if features is not None:
        result.append(np.concatenate(result_features))
    if labels is not None:
        result.append(np.concatenate(result_labels))
    return result


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None,
                                  sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=False):
    """Voxel barycentre subsampling, without random grid rotation."""
    points = _numpy(points, np.float32)
    lengths = _numpy(batches_len, np.int32)
    features = None if features is None else _numpy(features, np.float32)
    labels = None if labels is None else _numpy(labels, np.int32)
    if sampleDl <= 0:
        raise ValueError("sampleDl must be positive.")
    if random_grid_orient:
        raise ValueError("Random grid orientation is not used by this encoder.")
    if cpp_subsampling is not None:
        kwargs = dict(sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        if features is not None:
            kwargs["features"] = features
        if labels is not None:
            kwargs["classes"] = labels
        result = cpp_subsampling.subsample_batch(points, lengths, **kwargs)
    else:
        result = _voxel_subsample(points, lengths, features, labels, sampleDl, max_p)
    return tuple(torch.from_numpy(value) for value in result)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """Radius neighbours sorted by distance, padded by the support-count sentinel."""
    queries, supports = _numpy(queries, np.float32), _numpy(supports, np.float32)
    q_batches, s_batches = _numpy(q_batches, np.int32), _numpy(s_batches, np.int32)
    if cpp_neighbors is not None:
        neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
        if max_neighbors > 0:
            neighbors = neighbors[:, :max_neighbors]
        if neighbors.shape[1] == 0:
            neighbors = np.full((len(queries), 1), len(supports), dtype=np.int64)
        return torch.from_numpy(neighbors).long()

    from scipy.spatial import cKDTree
    rows, q_offset, s_offset = [], 0, 0
    for nq, ns in zip(q_batches, s_batches):
        nq, ns = int(nq), int(ns)
        cloud = supports[s_offset:s_offset + ns]
        query = queries[q_offset:q_offset + nq]
        tree = cKDTree(cloud)
        if max_neighbors > 0:
            k = min(max_neighbors, ns)
            distances, indices = tree.query(query, k=k, distance_upper_bound=radius)
            distances, indices = distances.reshape(nq, k), indices.reshape(nq, k)
            for distance, index in zip(distances, indices):
                valid = (index < ns) & (distance < radius)
                rows.append(index[valid] + s_offset)
        else:
            for point, indices in zip(query, tree.query_ball_point(query, radius)):
                index = np.asarray(indices, dtype=np.int64)
                distance2 = np.square(cloud[index] - point).sum(axis=1)
                index = index[np.argsort(distance2, kind="stable")]
                rows.append(index[np.square(cloud[index] - point).sum(axis=1) < radius ** 2] + s_offset)
        q_offset += nq
        s_offset += ns
    width = max(1, max((len(row) for row in rows), default=0))
    neighbors = np.full((len(queries), width), len(supports), dtype=np.int64)
    for i, row in enumerate(rows):
        neighbors[i, :len(row)] = row
    return torch.from_numpy(neighbors)


def input_package(pcd_list, config, neighborhood_limits, device=None):
    """Geometry is prepared on CPU and transferred to the encoder's device."""
    if not pcd_list:
        raise ValueError("pcd_list must contain at least one point cloud.")
    clouds = []
    for cloud in pcd_list:
        cloud = torch.as_tensor(cloud).detach().to(device="cpu", dtype=torch.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.shape[0] == 0:
            raise ValueError("Every point cloud must have shape (N, >=3), N > 0.")
        cloud = cloud[:, :3].contiguous()
        if not torch.isfinite(cloud).all():
            raise ValueError("Point clouds must contain finite coordinates.")
        clouds.append(cloud)
    points = torch.cat(clouds)
    lengths = torch.tensor([len(cloud) for cloud in clouds], dtype=torch.int32)
    features = torch.ones(len(points), config.in_feats_dim)
    inputs = {key: [] for key in ("points", "neighbors", "pools", "upsamples", "stack_lengths")}
    radius = config.first_subsampling_dl * config.conv_radius
    layer_blocks, layer = [], 0
    for block_i, block in enumerate(config.architecture):
        if "global" in block or "upsample" in block:
            break
        if not ("pool" in block or "strided" in block):
            layer_blocks.append(block)
            if block_i < len(config.architecture) - 1:
                continue
        conv_radius = radius
        if any("deformable" in item for item in layer_blocks[:-1]):
            conv_radius *= config.deform_radius / config.conv_radius
        conv = batch_neighbors_kpconv(points, points, lengths, lengths,
                                      conv_radius, neighborhood_limits[layer])
        if "pool" in block or "strided" in block:
            pool_points, pool_lengths = batch_grid_subsampling_kpconv(
                points, lengths, sampleDl=2 * radius / config.conv_radius)
            pool_radius = radius * (config.deform_radius / config.conv_radius if "deformable" in block else 1)
            pool = batch_neighbors_kpconv(pool_points, points, pool_lengths, lengths,
                                          pool_radius, neighborhood_limits[layer])
        else:
            pool = torch.zeros((0, 1), dtype=torch.long)
            pool_points = torch.zeros((0, 3), dtype=torch.float32)
            pool_lengths = torch.zeros(0, dtype=torch.int32)
        inputs["points"].append(points)
        inputs["neighbors"].append(conv)
        inputs["pools"].append(pool)
        # Encoder only: unused decoder upsampling neighbours are not computed.
        inputs["upsamples"].append(torch.zeros((0, 1), dtype=torch.long))
        inputs["stack_lengths"].append(lengths)
        points, lengths = pool_points, pool_lengths
        radius *= 2
        layer += 1
        layer_blocks = []
    inputs["features"] = features
    if device is not None:
        inputs = {key: ([item.to(device) for item in value] if isinstance(value, list)
                        else value.to(device)) for key, value in inputs.items()}
    return inputs


class KPFCN(nn.Module):
    """Four-level KPConv encoder; no unused decoder parameters."""
    def __init__(self):
        super().__init__()
        self.config = config = Config()
        layer = 0
        radius = config.first_subsampling_dl * config.conv_radius
        in_dim, out_dim = config.in_feats_dim, config.first_feats_dim
        self.encoder_blocks = nn.ModuleList()
        for block in config.architecture:
            self.encoder_blocks.append(block_decider(block, radius, in_dim, out_dim, layer, config))
            in_dim = out_dim // 2 if "simple" in block else out_dim
            if "pool" in block or "strided" in block:
                layer += 1
                radius *= 2
                out_dim *= 2

    def forward(self, pcd_list):
        device = next(self.parameters()).device
        inputs = input_package(pcd_list, self.config, self.config.neighborhood_limits, device=device)
        features = inputs["features"]
        for block in self.encoder_blocks:
            features = block(features, inputs)
        lengths = inputs["stack_lengths"][-1].tolist()
        return list(features.split(lengths)), list(inputs["points"][-1].split(lengths))
