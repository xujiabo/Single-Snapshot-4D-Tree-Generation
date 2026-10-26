"""轨迹解码、pipe model 和无需图形窗口的结果导出。"""
from pathlib import Path
import json
import numpy as np
from tree.species_presets import SPECIES_PRESETS

SPECIES_NAMES = ("oak", "birch", "maple", "pine", "willow")
# Species presets use the original coordinate units of the input point cloud.
PIPE_PRESETS = {name: (SPECIES_PRESETS[name]["LEAF_RADIUS"],
                       SPECIES_PRESETS[name]["PIPE_EXPONENT"]) for name in SPECIES_NAMES}
PIPE_PRESETS[None] = (0.003, 2.5)


def validate_tree(coords, parents, birth):
    coords, parents, birth = np.asarray(coords), np.asarray(parents), np.asarray(birth)
    n = len(coords)
    if n < 1 or coords.shape != (n, 3) or parents.shape != (n,) or birth.shape != (n,):
        raise ValueError("Tree arrays must be coords (N,3), parents (N,), birth (N,).")
    if not np.isfinite(coords).all() or parents[0] != -1:
        raise ValueError("Non-finite coordinates or missing root at index 0.")
    if not np.equal(parents, np.floor(parents)).all() or not np.equal(birth, np.floor(birth)).all():
        raise ValueError("Parents and birth steps must be integers.")
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, n)):
        raise ValueError("Every non-root parent must precede its child.")
    if np.any(birth < 1) or np.any(birth[1:] < birth[parents[1:].astype(int)]):
        raise ValueError("Birth steps must be positive and parent-before-child.")


def pipe_radii(parents, leaf_radius=0.003, exponent=2.5):
    """无chain平滑：严格保持 r_parent**eta = sum(r_child**eta)。"""
    parents = np.asarray(parents, dtype=np.int64)
    if not len(parents):
        return np.empty(0, np.float32)
    if leaf_radius <= 0 or exponent <= 0:
        raise ValueError("Pipe parameters must be positive.")
    counts = np.zeros(len(parents), np.float64)
    for i in range(len(parents) - 1, -1, -1):
        if counts[i] == 0:
            counts[i] = 1
        if parents[i] >= 0:
            counts[parents[i]] += counts[i]
    return (leaf_radius * counts ** (1.0 / exponent)).astype(np.float32)


def growth_frames(coords, parents, birth, max_step=35, species=None):
    """完整node_id保留在每一帧中；parents是该帧内重新编号的父指针。"""
    coords = np.asarray(coords, np.float32)
    parents, birth = np.asarray(parents, np.int64), np.asarray(birth, np.int64)
    validate_tree(coords, parents, birth)
    if max_step < int(birth.max()):
        raise ValueError("max_step is smaller than a predicted birth step.")
    leaf_radius, exponent = PIPE_PRESETS[species]
    frames = []
    for step in range(1, max_step + 1):
        ids = np.flatnonzero(birth <= step)
        index = np.full(len(coords), -1, dtype=np.int64)
        index[ids] = np.arange(len(ids))
        par = parents[ids].copy()
        nonroot = par >= 0
        par[nonroot] = index[par[nonroot]]
        frames.append(dict(step=step, node_ids=ids, coords=coords[ids], parents=par,
                           birth=birth[ids], radii=pipe_radii(par, leaf_radius, exponent)))
    return frames


def match_observed_frame(coords_normalized, birth, max_step=35):
    """argmin_k |max(z_i: birth_i<=k)-1|；空帧排除，同高时取最早帧。"""
    heights = np.full(max_step, np.nan, np.float64)
    for k in range(1, max_step + 1):
        mask = np.asarray(birth) <= k
        if mask.any():
            heights[k - 1] = np.asarray(coords_normalized)[mask, 2].max()
    if not np.isfinite(heights).any():
        raise ValueError("There is no non-empty growth frame to match.")
    errors = np.abs(heights - 1.0)
    best = int(np.nanargmin(errors))
    ties = np.flatnonzero(np.isclose(errors, errors[best], atol=1e-7, rtol=0)) + 1
    return dict(step=best + 1, height=float(heights[best]), error=float(errors[best]),
                tied_steps=ties.tolist(), heights=[float(v) if np.isfinite(v) else None for v in heights])


def nodes_from_arrays(coords, parents, birth, radii):
    return [dict(xyz=coords[i].copy(),
                 start_xyz=coords[i].copy() if p < 0 else coords[p].copy(),
                 parent_idx=int(p), birth_step=int(birth[i]), radius=float(radii[i]))
            for i, p in enumerate(parents)]


def save_graph_ply(path, coords, parents):
    """ASCII PLY 带edge，MeshLab等可直接读取。"""
    edge = [(int(p), i) for i, p in enumerate(parents) if p >= 0]
    with Path(path).open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(coords))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element edge %d\nproperty int vertex1\nproperty int vertex2\nend_header\n" % len(edge))
        for row in coords:
            f.write("%.8g %.8g %.8g\n" % tuple(row))
        for a, b in edge:
            f.write("%d %d\n" % (a, b))


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def save_preview(frames, cloud, matched_step, path, save_gif=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image

    all_xyz = np.concatenate([cloud, frames[-1]["coords"]], 0)
    low, high = all_xyz.min(0), all_xyz.max(0)
    center, half = (low + high) / 2, max(float((high - low).max()) / 2, 1e-4) * 1.05

    def draw(ax, frame):
        xyz, par = frame["coords"], frame["parents"]
        ids = np.flatnonzero(par >= 0)
        if len(ids):
            lines = np.stack([xyz[par[ids]], xyz[ids]], 1)
            ax.add_collection3d(Line3DCollection(lines, colors="#654126", linewidths=0.8))
        elif len(xyz):
            ax.scatter(*xyz.T, s=10, c="#654126")
        pc = cloud[::max(1, len(cloud) // 1500)]
        ax.scatter(*pc.T, s=0.5, c="#2196d2", alpha=0.14)
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               zlim=(center[2]-half, center[2]+half))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-70)
        ax.set_axis_off()
        ax.set_title("Step %d%s" % (frame["step"], " (input match)" if frame["step"] == matched_step else ""))

    selected = sorted(set(np.linspace(1, len(frames), 6).round().astype(int).tolist() + [matched_step]))
    fig = plt.figure(figsize=(3 * len(selected), 4))
    for i, k in enumerate(selected):
        draw(fig.add_subplot(1, len(selected), i+1, projection="3d"), frames[k-1])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    if save_gif:
        images = []
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        for frame in frames:
            ax.clear()
            draw(ax, frame)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            images.append(Image.fromarray(rgba[:, :, :3].copy()))
        plt.close(fig)
        images[0].save(str(Path(path).with_suffix(".gif")), save_all=True,
                       append_images=images[1:], duration=180, loop=0)
