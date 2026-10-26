"""Generate and export a growth trajectory using settings from config.py."""
from pathlib import Path
import numpy as np
import torch
from config import INFER, DATA
from dataset import TreeDataset, prepare_point_cloud, renormalize_pc_with_meta
from model import TreeFlow
from train import choose_device
from utils import (SPECIES_NAMES, growth_frames, match_observed_frame, save_graph_ply,
                   save_preview, save_json, nodes_from_arrays)


def read_point_cloud(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
        if points.ndim == 2 and points.shape[1] == 9:
            raise ValueError("This looks like a 9-column L-system tree, not an input scan. Use a point-cloud file, or point_cloud=None for the simulated demo.")
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((k for k in ("points", "xyz", "pcd", "point_cloud") if k in archive), None)
            if key is None:
                raise ValueError("NPZ must contain points, xyz, pcd, or point_cloud.")
            points = archive[key]
    elif suffix in (".txt", ".xyz", ".csv"):
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None, ndmin=2)
    elif suffix in (".ply", ".pcd", ".pts", ".xyzn", ".xyzrgb"):
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        raise ValueError("Unsupported point cloud format: %s" % suffix)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected an N x 3 (or XYZ plus attributes) point cloud.")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        raise ValueError("At least two finite point-cloud points are required.")
    return points


def load_model(checkpoint, device):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: %s. Train with train.py first; smoke_test.py only verifies the pipeline." % path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model_config" not in state or "model" not in state:
        raise ValueError("Use a checkpoint saved by this project's train.py with model_config and model entries.")
    if state.get("normalization") != "input_bbox_top1_isotropic_v1":
        raise ValueError("Checkpoint uses incompatible normalization.")
    net = TreeFlow(**state["model_config"]).to(device)
    net.load_state_dict(state["model"], strict=True)
    net.eval()
    return net, state


def export_prediction(prediction, cloud_normalized, meta, output_dir, species=None,
                      max_step=35, export_mesh=True, save_gif=True, extra=None):
    """输入是单棵树N维数组；所有导出坐标还原到输入点云坐标系。"""
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    coords_n = array(prediction["coords"]).astype(np.float32)
    parents = array(prediction["parents"]).astype(np.int64)
    birth = array(prediction["birth"]).astype(np.int64)
    birth_raw = array(prediction.get("birth_raw", birth)).astype(np.int64)
    coords = renormalize_pc_with_meta(coords_n, meta)
    cloud = renormalize_pc_with_meta(np.asarray(cloud_normalized), meta)
    match = match_observed_frame(coords_n, birth, max_step)
    frames = growth_frames(coords, parents, birth, max_step, species)
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "trajectory.npz", coords=coords, coords_normalized=coords_n,
                        parents=parents, birth=birth, birth_raw=birth_raw,
                        radii=frames[-1]["radii"], point_cloud=cloud,
                        point_cloud_normalized=cloud_normalized,
                        active=np.stack([birth <= k for k in range(1, max_step+1)]),
                        matched_step=match["step"])
    for frame in frames:
        np.savez_compressed(frame_dir / ("step_%02d.npz" % frame["step"]), **frame)
    matched = frames[match["step"]-1]
    np.savez_compressed(output_dir / "matched_frame.npz", **matched)
    save_graph_ply(output_dir / "full_tree.ply", coords, parents)
    save_graph_ply(output_dir / "matched_frame.ply", matched["coords"], matched["parents"])
    if export_mesh:
        import open3d as o3d
        from tree.tree3D_utils import build_tree_mesh
        for name, frame in (("full_tree_mesh", frames[-1]), ("matched_frame_mesh", matched)):
            if len(frame["coords"]) > 1:
                nodes = nodes_from_arrays(frame["coords"], frame["parents"], frame["birth"], frame["radii"])
                mesh = build_tree_mesh(nodes, radial_segments=8)
                if not o3d.io.write_triangle_mesh(str(output_dir / (name + ".ply")), mesh):
                    raise IOError("Failed to save %s" % name)
    save_json(output_dir / "metadata.json", dict(
        normalization=meta, species=species, match=match, max_step=max_step,
        frame_rule="birth <= k; parents reindexed per frame; node_ids retain complete-tree correspondence",
        match_rule="min abs(max active normalized z - 1); earliest step wins ties",
        radius_units="same as the input point cloud; species presets assume the source dataset unit",
        extra=extra or {}))
    save_preview(frames, cloud, match["step"], output_dir / "trajectory.png", save_gif)
    print("Input match: step %d | normalized top %.5f | |top-1| %.5f" %
          (match["step"], match["height"], match["error"]), flush=True)
    if len(match["tied_steps"]) > 1:
        print("Equal-height candidates: %s (earliest selected)." % match["tied_steps"])
    print("Saved trajectory to: %s" % output_dir, flush=True)
    return dict(output_dir=output_dir, match=match, frames=frames)


@torch.no_grad()
def infer(settings=None):
    settings = dict(INFER if settings is None else settings)
    device = choose_device(settings["device"])
    net, state = load_model(settings["checkpoint"], device)
    species = settings["species"]
    if species is not None and species not in SPECIES_NAMES:
        raise ValueError("species must be one of %s or None." % (SPECIES_NAMES,))
    if settings["point_cloud"] is None:
        data = dict(DATA)
        # 数据可移动到新机器，但预处理参数跟随训练checkpoint。
        data["voxel_size"] = state.get("data_config", {}).get("voxel_size", .025)
        data.update(n_nodes=net.N, n_birth=state["model_config"]["n_birth"], max_points=settings["max_points"])
        sample = TreeDataset(**data, split="all", augment=False)[0]
        cloud_n = np.asarray(sample["pcd"], dtype=np.float32)
        meta = sample["meta"]
        source = "dataset simulated scan (unknown species unless configured)"
    else:
        cloud = read_point_cloud(settings["point_cloud"])
        cloud_n, meta = prepare_point_cloud(
            cloud, max_points=settings["max_points"], seed=settings["seed"],
            voxel_size=state.get("data_config", {}).get("voxel_size", .025))
        source = str(settings["point_cloud"])
    sp = None if species is None else SPECIES_NAMES.index(species)
    result = net.sample(species=sp, pcd_list=[torch.from_numpy(cloud_n)], steps=settings["steps"],
                        guidance=settings["guidance"], seed=settings["seed"])
    prediction = {k: v[0] for k, v in result.items()}
    return export_prediction(prediction, cloud_n, meta, settings["output_dir"], species,
                             state["model_config"]["n_birth"], settings["export_mesh"], settings["save_gif"],
                             extra=dict(source=source, checkpoint=str(settings["checkpoint"]),
                                        epoch=state.get("epoch"), steps=settings["steps"],
                                        guidance=settings["guidance"], seed=settings["seed"]))


if __name__ == "__main__":
    infer()
