"""Sequential train/validation/test lifecycle shared by all methods."""
import copy
import csv
import json
from pathlib import Path
import random
import platform
import shutil
import time
import numpy as np
import torch
from data.manifest import validate_sequence
from data.replay import MixedBatchSampler
from methods.registry import create_method
from train_eval.metrics import TaskMatrix
from train_eval.runtime import build_detector, build_dataset, dataloader, evaluate, prepare_batch


def load_initial(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload))
    clean = {key[6:] if key.startswith("model.") else key: value for key, value in state.items() if not key.startswith(("teacher.", "avg_model."))}
    # Strict loading prevents silently replacing a 12-class head with 13-class weights.
    model.load_state_dict(clean, strict=True)


def run(config, method_name, device="cuda:0"):
    categories, data_report = validate_sequence(config)
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    project = config.get("profile", "baseline") == "project"
    if project and method_name != "REP":
        raise ValueError("Project profile uses hierarchical REP; run ten baselines with profile: baseline")
    if method_name == "KMREP" and not Path(config.get("resnet_weights") or "").is_file():
        raise FileNotFoundError("KMREP requires local resnet_weights")
    config = copy.deepcopy(config)
    run_dir = Path(config["output"]) / f"{config.get('profile', 'baseline')}_{method_name}"
    # New runs never overwrite checkpoints or mix metrics from different experiments.
    run_dir.mkdir(parents=True, exist_ok=False)
    config["run_dir"] = str(run_dir)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "data_audit.json").write_text(json.dumps(data_report, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(config["detector"], run_dir / "detector.yml")
    (run_dir / "runtime.json").write_text(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
                                                    "device": device, "cuda_build": torch.version.cuda,
                                                    "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None}, indent=2), encoding="utf-8")
    model = build_detector(config, len(categories))
    if config.get("initial_checkpoint"):
        load_initial(model, config["initial_checkpoint"])
    if project:
        from model.experts import SharedExpertDetector
        model = SharedExpertDetector(model, config.get("experts", {}).get("count", 3), config.get("experts", {}).get("router_weight", 1.0))
        from methods.project import ProjectReplay
        strategy = ProjectReplay(config)
    else:
        strategy = create_method(method_name, config)
    model.to(device)
    matrix = TaskMatrix([task["name"] for task in config["tasks"]])
    validation_sets, test_sets = [], []
    try:
        for index, task in enumerate(config["tasks"]):
            stage = run_dir / f"task_{index + 1:02d}_{task['name']}"
            stage.mkdir()
            current = build_dataset(config, task, "train", len(categories))
            validation_sets.append(build_dataset(config, task, "val", len(categories)))
            test_sets.append(build_dataset(config, task, "test", len(categories)))
            strategy.before_task(model, index)
            if project:
                model.select_training_expert(task.get("expert", 0), freeze_shared=index > 0)
            selected = strategy.train_dataset(current)
            if strategy.uses_replay and len(strategy.buffer):
                selected = torch.utils.data.ConcatDataset([current, strategy.buffer])
                sampler = MixedBatchSampler(len(current), len(strategy.buffer), config["training"]["batch_size"], seed + index,
                                            config.get("replay", {}).get("ratio", 0.5))
                loader = dataloader(selected, config, sampler=sampler)
            else:
                loader = dataloader(selected, config, shuffle=True)
            train_stage(model, strategy, loader, validation_sets, config, device, stage)
            strategy.close()
            strategy.after_task(model, current, dataloader(current, config), device)
            torch.save({"model": model.state_dict(), "categories": categories, "task": index,
                        "profile": config.get("profile", "baseline"), "experts": config.get("experts", {})}, stage / "model.pt")
            if project:
                strategy.memory.archive(stage / "model.pt", f"task-{index + 1:02d}", categories)
            for tested, dataset in enumerate(test_sets):
                result = evaluate(model, dataset, config, device, stage / "test" / config["tasks"][tested]["name"])
                matrix.set(index, tested, result["mAP"])
            matrix.save(run_dir)
            print(f"{method_name} task {index + 1}: {matrix.summary(index)}", flush=True)
    finally:
        strategy.close()
    return run_dir


def train_stage(model, strategy, loader, validation_sets, config, device, stage):
    training = config["training"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=training.get("lr", .001), weight_decay=training.get("weight_decay", .05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training["epochs"], eta_min=training.get("min_lr", .00005))
    best, best_state = -float("inf"), None
    with (stage / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "step", "loss", "lr", "elapsed_seconds", "loss_components_json"])
        start = time.monotonic()
        for epoch in range(training["epochs"]):
            model.train()
            model.set_epoch(epoch)
            strategy.prepare_mode(model)
            for step, batch in enumerate(loader):
                batch = prepare_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, states = strategy.loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite detection/continual loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("grad_clip", 35))
                optimizer.step()
                components = {name: float(value.detach().mean()) if torch.is_tensor(value) else float(value) for name, value in states.items()}
                writer.writerow([epoch + 1, step, loss.detach().item(), optimizer.param_groups[0]["lr"], time.monotonic() - start, json.dumps(components)])
            scheduler.step()
            stream.flush()
            if (epoch + 1) % training["val_interval"] == 0 or epoch + 1 == training["epochs"]:
                # The fixed validation sets of all observed tasks select the release.
                scores = [evaluate(model, dataset, config, device, stage / "validation" / f"task_{i + 1:02d}")["mAP"] for i, dataset in enumerate(validation_sets)]
                average = sum(scores) / len(scores)
                if average > best:
                    best = average
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    torch.save({"model": best_state, "epoch": epoch + 1, "validation_mAP": best}, stage / "best_validation.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}, stage / "last_epoch.pt")
    model.load_state_dict(best_state)
    maintenance = config.get("maintenance", {})
    if maintenance.get("enabled", False):
        from maintenance.microsleep import maintain
        def fixed_validation(network):
            scores = [evaluate(network, dataset, config, device, stage / "maintenance_validation" / f"task_{i + 1:02d}")["mAP"] for i, dataset in enumerate(validation_sets)]
            return sum(scores) / len(scores)
        result = maintain(model, maintenance["threshold"], maintenance["budget_seconds"], fixed_validation,
                          maintenance.get("allowed_drop", 0), maintenance.get("protected", []))
        (stage / "maintenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
