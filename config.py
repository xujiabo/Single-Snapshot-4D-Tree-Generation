"""Edit these settings, then run train.py, infer.py, or smoke_test.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL = dict(d=384, nhead=6, layers=12, n_species=5, n_nodes=1024,
             pcd_token_num=256, n_birth=35, pc_layers=4)

DATA = dict(root=ROOT / "L-system-generated", n_nodes=1024, n_birth=35,
            max_points=4096, cache_dir=ROOT / "dataset_cache", seed=42,
            val_fraction=0.1)

TRAIN = dict(
    epochs=200, batch_size=4, num_workers=0, lr=5e-4, min_lr=1e-5,
    warmup_epochs=5, tau=0.98, sigma_max=0.04, lambda_birth=1.0,
    p_drop_pc=0.1, p_drop_species=0.1, grad_clip=1.0,
    device="cuda:0", seed=42, save_dir=ROOT / "params",
    resume=None,                 # Resume: ROOT / "params" / "last.pth"
    max_batches=None,            # Optional debug limit, e.g. 2
    val_batches=4,               # Loss/head diagnostics, not full paper evaluation
    save_every=10,
)

INFER = dict(
    checkpoint=ROOT / "params" / "last.pth",  # Train first; no weights are bundled
    point_cloud=None,            # A point-cloud path; None uses a simulated example
    species=None,                # oak/birch/maple/pine/willow; None = unknown
    steps=25, guidance=2.0,       # v = v_null + (1+w)*(v_cond-v_null); w=0 is conditional
    seed=42, device="cuda:0", max_points=4096,
    output_dir=ROOT / "predictions", export_mesh=True, save_gif=True,
)
