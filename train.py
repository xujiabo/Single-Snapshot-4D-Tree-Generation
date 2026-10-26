"""Train the shared model using settings from config.py."""
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import MODEL, DATA, TRAIN
from dataset import TreeDataset, collate
from model import TreeFlow
from utils import save_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def model_loss(net, batch, device, settings, regime="mixed", training=True):
    return net.loss(
        batch["species"].to(device), batch["coords"].to(device),
        batch["parents"].to(device), batch["birth"].to(device), batch["pcd_list"],
        tau=settings["tau"], sigma_max=settings["sigma_max"],
        lambda_birth=settings["lambda_birth"],
        p_drop_pc=settings["p_drop_pc"] if training else 0.0,
        p_drop_species=settings["p_drop_species"] if training else 0.0, regime=regime)


def learning_rate(epoch, settings):
    warm = min(settings["warmup_epochs"], settings["epochs"])
    if warm and epoch <= warm:
        return settings["lr"] * epoch / warm
    progress = (epoch - warm) / max(1, settings["epochs"] - warm)
    return settings["min_lr"] + (settings["lr"] - settings["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def validate(net, loader, device, settings):
    """教师坐标下结构分类诊断 + 独立流损失；不是生成轨迹精度。"""
    net.eval()
    totals, count = {}, 0
    # 不让验证过程改变下一epoch训练的随机流；每次验证噪声也一致。
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2026)
        for i, batch in enumerate(loader):
            if settings["val_batches"] is not None and i >= settings["val_batches"]:
                break
            flow = model_loss(net, batch, device, settings, "flow", False)
            terminal = model_loss(net, batch, device, settings, "terminal", False)
            n = len(batch["species"])
            metrics = {"flow": flow["flow"], "parent": terminal["parent"], "birth": terminal["birth"],
                       "parent_acc": terminal["parent_acc"], "birth_acc": terminal["birth_acc"]}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * n
            count += n
    return {k: v / count for k, v in totals.items()} if count else {}


def save_checkpoint(path, net, optimizer, epoch, model_config, data_config, settings, history):
    state = dict(format_version=1, model=net.state_dict(), optimizer=optimizer.state_dict(),
                 epoch=epoch, model_config=dict(model_config), data_config=dict(data_config),
                 train_config=dict(settings), history=history,
                 normalization="input_bbox_top1_isotropic_v1",
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def train(model_config=None, data_config=None, settings=None):
    model_config = dict(MODEL if model_config is None else model_config)
    data_config = dict(DATA if data_config is None else data_config)
    settings = dict(TRAIN if settings is None else settings)
    if data_config["n_nodes"] != model_config["n_nodes"] or data_config["n_birth"] != model_config["n_birth"]:
        raise ValueError("MODEL and DATA node/birth counts must agree.")
    seed_all(settings["seed"])
    device = choose_device(settings["device"])
    train_set = TreeDataset(**data_config, split="train", augment=True)
    val_set = TreeDataset(**data_config, split="val", augment=False)
    loader = DataLoader(train_set, batch_size=settings["batch_size"], shuffle=True,
                        num_workers=settings["num_workers"], collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=settings["batch_size"], shuffle=False,
                            num_workers=settings["num_workers"], collate_fn=collate)
    net = TreeFlow(**model_config).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"])
    start, history = 1, []
    if settings["resume"]:
        state = torch.load(settings["resume"], map_location="cpu", weights_only=False)
        if state["model_config"] != model_config or state.get("normalization") != "input_bbox_top1_isotropic_v1":
            raise ValueError("Resume checkpoint model/normalization is incompatible with config.py.")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["epoch"] + 1, state["history"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_dir = Path(settings["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Device: %s | train/val trees: %d/%d | parameters: %.2fM" %
          (device, len(train_set), len(val_set), sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
    print("First access builds LiDAR scan cache. Mixed objective: %.1f%% flow / %.1f%% terminal." %
          (100 * settings["tau"], 100 * (1-settings["tau"])), flush=True)
    for epoch in range(start, settings["epochs"] + 1):
        net.train()
        lr = learning_rate(epoch, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        total, count, terminal_count = 0.0, 0, 0
        tic = time.time()
        for i, batch in enumerate(loader):
            if settings["max_batches"] is not None and i >= settings["max_batches"]:
                break
            optimizer.zero_grad(set_to_none=True)
            result = model_loss(net, batch, device, settings)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss at epoch %d batch %d" % (epoch, i))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            n = len(batch["species"])
            total += loss.item() * n
            count += n
            terminal_count += int(result["terminal_count"])
            print("\rEpoch %d/%d  batch %d/%d  loss %.5f  terminal %d  lr %.6f" %
                  (epoch, settings["epochs"], i+1, len(loader), loss.item(), terminal_count, lr), end="", flush=True)
        if not count:
            raise ValueError("No training batch was processed. Check data and max_batches.")
        metrics = validate(net, val_loader, device, settings)
        row = dict(epoch=epoch, train_loss=total/count, terminal_samples=terminal_count,
                   seconds=time.time()-tic, lr=lr, validation=metrics)
        history.append(row)
        print("\nMean %.5f | val %s | %.1fs" % (row["train_loss"], metrics, row["seconds"]), flush=True)
        save_checkpoint(save_dir / "last.pth", net, optimizer, epoch, model_config, data_config, settings, history)
        if epoch % settings["save_every"] == 0:
            save_checkpoint(save_dir / ("epoch-%03d.pth" % epoch), net, optimizer, epoch, model_config, data_config, settings, history)
        save_json(save_dir / "history.json", history)
    return net, history


if __name__ == "__main__":
    train()
