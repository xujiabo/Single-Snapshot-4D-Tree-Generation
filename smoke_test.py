"""一键检查真实数据->训练反传->保存/重载->点云推理->35帧导出，不训练成品模型。"""
import gc
import time
import numpy as np
import torch
from config import ROOT, MODEL, DATA, TRAIN, INFER
from dataset import TreeDataset, collate, normalize_point_cloud, normalize_pc_with_meta, renormalize_pc_with_meta
from train import train, model_loss, choose_device, save_checkpoint
from infer import infer
from utils import growth_frames, match_observed_frame, pipe_radii, validate_tree, save_json

# CUDA时检查论文完整骨干/1024节点。CPU使用小骨干，但仍是相同的KPConv代码。
FULL_MODEL = torch.cuda.is_available()
# 测试checkpoint含Adam状态会很大；默认验证重载后删除，避免误作训练成品。
KEEP_TEST_CHECKPOINTS = False


def check_invariants():
    # 宽冠也要同时保证各轴范围和top=1；外推目标不可截断；平移/缩放可逆。
    cloud = np.array([[-4., -2., 5.], [4., 2., 7.], [0., 0., 6.]], np.float32)
    pc, meta = normalize_point_cloud(cloud)
    assert abs(pc[:, 2].max() - 1) < 1e-6 and np.abs(pc).max() <= 1.000001
    np.testing.assert_allclose(renormalize_pc_with_meta(pc, meta), cloud, atol=1e-6)
    future = normalize_pc_with_meta(np.array([[0., 0., 10.]], np.float32), meta)
    assert future[0, 2] > 1
    # 明确的跨越1案例：必须取最高点0.95那帧，不是终态或第一个越过1的帧。
    xyz = np.array([[0, 0, -1], [0, 0, .5], [0, 0, .95], [0, 0, 1.4]], np.float32)
    parents, birth = np.array([-1, 0, 1, 1]), np.array([1, 2, 3, 4])
    match = match_observed_frame(xyz, birth, 5)
    assert match["step"] == 3
    frames = growth_frames(xyz, parents, birth, 5, "oak")
    for before, after in zip(frames, frames[1:]):
        assert set(before["node_ids"]).issubset(after["node_ids"])
    radius = pipe_radii(parents, .005, 2.8)
    np.testing.assert_allclose(radius[1] ** 2.8, radius[2] ** 2.8 + radius[3] ** 2.8, rtol=1e-6)


def main():
    started = time.time()
    check_invariants()
    out = ROOT / "work" / "smoke_test"
    out.mkdir(parents=True, exist_ok=True)
    model_config = dict(MODEL)
    if not FULL_MODEL:
        model_config.update(d=48, nhead=4, layers=2, n_nodes=64, pcd_token_num=16, pc_layers=1)
    data_config = dict(DATA, n_nodes=model_config["n_nodes"], max_points=1024,
                       max_trees=2, scan_fractions=(.7,), scan_configs=(("tls", "sparse"),))
    settings = dict(TRAIN, epochs=1, batch_size=1, max_batches=1, val_batches=1,
                    save_dir=out / "checkpoints", resume=None)
    device = choose_device(settings["device"])
    print("Smoke test: %s, full model=%s, node slots=%d" % (device, FULL_MODEL, model_config["n_nodes"]), flush=True)
    net, history = train(model_config, data_config, settings)
    dataset = TreeDataset(**data_config, augment=False, split="all")
    sample = dataset[0]
    batch = collate([sample])
    assert abs(sample["pcd"][:, 2].max()-1) < 1e-6
    assert np.abs(sample["pcd"]).max() <= 1.000001
    validate_tree(sample["coords"], sample["parents"], sample["birth"])
    assert sample["coords"][:, 2].max() > 1  # observed mid-growth scan, full target extends above it
    train_files = {p for p, _ in TreeDataset(**data_config, augment=False, split="train").files}
    val_files = {p for p, _ in TreeDataset(**data_config, augment=False, split="val").files}
    assert train_files.isdisjoint(val_files)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    diagnostics = {}
    for regime, heads in (("flow", ("vel_head",)), ("terminal", ("q_proj", "k_proj", "birth_head"))):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = model_loss(net, batch, device, settings, regime)
        assert torch.isfinite(metrics["loss"])
        metrics["loss"].backward()
        for head_name in heads:
            grad = getattr(net, head_name).weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, head_name
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        diagnostics[regime] = {k: float(v) for k, v in metrics.items()}
        print("%s backward: %s" % (regime, diagnostics[regime]), flush=True)

    checkpoint = out / "checkpoints" / "pipeline_only.pth"
    save_checkpoint(checkpoint, net, optimizer, 0, model_config, data_config, settings, history)
    # 保存原始坐标下的真实点云，再通过外部文件入口读取并重新归一化。
    cloud_path = out / "input_scan.npy"
    np.save(cloud_path, renormalize_pc_with_meta(sample["pcd"], sample["meta"]))
    del net, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    inference = dict(INFER, checkpoint=checkpoint, point_cloud=cloud_path, max_points=1024,
                     species="oak", steps=25, output_dir=out / "prediction", export_mesh=True, save_gif=True)
    result = infer(inference)
    with np.load(result["output_dir"] / "trajectory.npz") as trajectory:
        validate_tree(trajectory["coords"], trajectory["parents"], trajectory["birth"])
        assert trajectory["active"].shape == (35, model_config["n_nodes"])
        assert np.all(trajectory["active"][:-1] <= trajectory["active"][1:])
        assert trajectory["active"][-1].all()
    assert len(list((result["output_dir"] / "frames").glob("*.npz"))) == 35
    report = dict(passed=True, full_model=FULL_MODEL, model=model_config, device=str(device),
                  torch=torch.__version__, seconds=time.time()-started,
                  training_history=history, branch_diagnostics=diagnostics,
                  matching=result["match"], note="Pipeline checks only; these weights are NOT a trained tree generator.")
    save_json(out / "report.json", report)
    if not KEEP_TEST_CHECKPOINTS:
        for path in (checkpoint, out / "checkpoints" / "last.pth"):
            if path.is_file():
                path.unlink()
    print("PASS: train, both heads, checkpoint reload, 25 Euler steps, frame matching, 35-frame export.")
    print("Smoke artifacts: %s (untrained pipeline output; not a quality demonstration)." % out)
    print("Elapsed %.1f seconds" % report["seconds"])
    return report


if __name__ == "__main__":
    main()
