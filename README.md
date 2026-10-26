# Single-Snapshot 4DTree

**Single-Snapshot 4D Tree Generation via Unified Continuous–Discrete Flow Matching**

<table>
  <tr>
    <td align="center"><img src="assets/birch_1.gif" height="240" alt="Birch growth example 1"><br>Birch</td>
    <td align="center"><img src="assets/birch_2.gif" height="240" alt="Birch growth example 2"><br>Birch</td>
    <td align="center"><img src="assets/maple_1.gif" height="240" alt="Maple growth example"><br>Maple</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/oak_1.gif" height="240" alt="Oak growth example"><br>Oak</td>
    <td align="center"><img src="assets/pine_1.gif" height="240" alt="Pine growth example"><br>Pine</td>
    <td align="center"><img src="assets/willow_1.gif" height="240" alt="Willow growth example"><br>Willow</td>
  </tr>
</table>

*Tree growth examples from our project. The current inference script exports skeleton and branch-mesh previews; the full foliage-rendering pipeline is part of the planned release.*

## Release status

**The paper is currently under review. The full training dataset and pretrained model weights will be released in the future. The remaining code is coming soon.**

This preview contains the core model, data processing, training and inference scripts, synthetic example data, and the animations above. No trained weights, cached scans, or previous experiment outputs are included. The examples support testing and small-scale training; they are not the full dataset used for the paper. The remaining code and complete experiment/evaluation materials will be added in a future release.

## Overview

Given one point cloud of a tree at an unknown growth stage, the model generates a complete skeleton with node coordinates, parent pointers, and per-node birth steps. A shared transformer performs continuous coordinate flow matching and terminal discrete decoding. Thresholding the birth steps produces a sequence of up to 35 nested tree stages, while branch radii are derived from the topology using the pipe model.

## Installation

Run commands from the repository root in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation for GPU training. The code has been checked with Python 3.8.3, PyTorch 2.4.1+cu118, NumPy 1.24.4, SciPy 1.10.1, and Open3D 0.17.0 on an NVIDIA RTX 4090. Other environment combinations have not been validated for this preview.

Precompiled native binaries are intentionally omitted. KPConv automatically uses the included NumPy/SciPy geometry preprocessing when its optional C++ extensions are unavailable; the learned network remains the same. A warning announcing this backend is expected. Optional compilation instructions are in [models/kpconv/cpp_wrappers/README.md](models/kpconv/cpp_wrappers/README.md).

All settings live in [`config.py`](config.py). The scripts do not require command-line flags and can also be run directly from an IDE. Project paths are resolved relative to `config.py`.

## Quick start

Inspect one example and generate its simulated scan:

```bash
python dataset.py
```

Verify the pipeline without downloading weights:

```bash
python smoke_test.py
```

The smoke test performs a few optimization steps, checks both prediction branches, saves and reloads a temporary checkpoint, and exports a 35-frame trajectory to `work/smoke_test/`. It uses the full model on CUDA and a smaller model on CPU. Temporary checkpoints are deleted after a successful test by default. These briefly optimized weights and their outputs do not represent trained-model quality.

## Training

Edit the following settings in `config.py`:

```python
DATA["root"] = ROOT / "L-system-generated"
TRAIN["device"] = "cuda:0"
TRAIN["batch_size"] = 4
TRAIN["epochs"] = 200
TRAIN["resume"] = None
```

Then run:

```bash
python train.py
```

The loader splits entire trees by species into training and validation sets (90% / 10% by default), canonicalizes each tree to 1,024 nodes, and simulates a LiDAR scan at a sampled growth stage. First access is slower because scans are generated and cached in `dataset_cache/`.

Training writes `params/last.pth`, periodic checkpoints such as `params/epoch-010.pth`, and `params/history.json`. To resume, set `TRAIN["resume"] = ROOT / "params" / "last.pth"` and rerun `train.py`. Lower the batch size if GPU memory is limited. If CUDA is unavailable, the scripts fall back to CPU; full training is substantially slower there.

The default settings and bundled examples are intended to make the code usable. They do not reproduce the full paper experiment: the manuscript uses 10,000 training trees and batch size 20. The validation log contains loss/classification diagnostics rather than the paper's complete trajectory evaluation.

## Inference

**A compatible trained checkpoint is required.** Train one with `train.py`, or use the pretrained weights when they are released. Running `infer.py` before a checkpoint exists will report a missing-checkpoint error.

Set `INFER` in `config.py`, for example:

```python
INFER["checkpoint"] = ROOT / "params" / "last.pth"
INFER["point_cloud"] = ROOT / "inputs" / "tree.ply"  # your point-cloud file
INFER["species"] = "oak"  # or birch, maple, pine, willow, or None
INFER["output_dir"] = ROOT / "predictions"
```

```bash
python infer.py
```

Use an already segmented single-tree point cloud with the z axis pointing upward. The script applies scan-based normalization and exports coordinates in the input coordinate system. It does not estimate the ground or upright orientation. Radius presets assume the synthetic data's units; adjust the presets in `tree/species_presets.py` when using different units, or `PIPE_PRESETS[None]` in `utils.py` for an unknown species.

Supported inputs include `.ply`, `.pcd`, `.xyz`, `.txt`, `.csv`, `.npy`, and `.npz`. Text files should have no header, with XYZ in the first three columns. NumPy point clouds should have shape `(N, 3)`; an NPZ archive must contain a `points`, `xyz`, `pcd`, or `point_cloud` array. Set `species=None` for an unknown species. Set `point_cloud=None` to use a simulated scan from the examples; this still requires a checkpoint.

The nine-column files under `L-system-generated/` describe tree graphs, not input scans. Do not pass them directly as `INFER["point_cloud"]`.

Main outputs in `predictions/`:

| File | Contents |
| --- | --- |
| `trajectory.npz` | Complete tree, parents, birth steps, radii, input cloud, active-node masks, and matched step |
| `frames/step_01.npz` … `step_35.npz` | Per-stage tree graphs with local parents and original node IDs |
| `full_tree.ply`, `matched_frame.ply` | Complete and matched skeletons |
| `full_tree_mesh.ply`, `matched_frame_mesh.ply` | Branch meshes when mesh export is enabled |
| `trajectory.png`, `trajectory.gif` | Skeleton growth previews |
| `metadata.json` | Normalization, settings, and stage-matching information |

The matched stage uses a top-height heuristic, and multiple stages can share the same height. It is an approximate alignment rather than a guarantee of recovering the unique observation time.

## Example data

`L-system-generated/` contains 1,000 synthetic tree examples: 200 each for birch, maple, oak, pine, and willow. These are provided for testing the data pipeline and experimenting with training. The full training dataset will be released separately.

Each `.npy` file stores one `(N, 9)` array:

```text
[start_x, start_y, start_z, x, y, z, radius, birth_step, parent_idx]
```

The root is node 0 with parent `-1`; other parents precede their children. Birth steps are integer growth labels. The loader converts the raw trees to the configured node count. New data should follow the same species-directory structure and array format.

## Repository layout

```text
Single-Snapshot-4DTree/
├── assets/                # Growth animations shown above
├── L-system-generated/    # Synthetic example/test trees
├── models/                # KPConv encoder and point-cloud transformer
├── tree/                  # Tree geometry, species presets, LiDAR simulation
├── config.py              # Model, data, training, and inference settings
├── dataset.py             # Data loading, normalization, scan generation
├── model.py               # Unified continuous-discrete model
├── train.py               # Training and checkpointing
├── infer.py               # Inference and trajectory export
├── utils.py               # Topology, radii, and visualization utilities
├── smoke_test.py          # End-to-end pipeline check
└── requirements.txt
```

## Acknowledgments

The point-cloud encoder includes code adapted from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch). Original author notices and the bundled third-party license texts are retained; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
