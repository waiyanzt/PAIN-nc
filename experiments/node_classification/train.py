"""Train one PAIN IMDb node-classification run."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pain_nc.config import load_config, merged_config
from pain_nc.data import PainGraph, load_imdb_graph
from pain_nc.model import PainNodeClassifier
from pain_nc.telemetry import (
    artifact_sizes,
    cuda_memory_stats,
    environment_metadata,
    model_memory_bytes,
    PeakRSSMonitor,
    reset_cuda_peak,
    serialized_torch_bytes,
    validate_resource_metrics,
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Configuration requests {requested!r}, but CUDA is unavailable. "
            "Use --device cpu for a CPU run."
        )
    return torch.device(requested)


def build_model(graph: PainGraph, config: dict[str, Any]) -> PainNodeClassifier:
    model_config = dict(config["model"])
    model_config.pop("reverse_paths", None)
    model_config.pop("paths_on_device", None)
    return PainNodeClassifier(
        input_dim=graph.num_features,
        num_classes=graph.num_classes,
        num_node_types=graph.num_node_types,
        num_edge_types=graph.num_edge_types,
        **model_config,
    )


@torch.no_grad()
def masked_validation_metrics(
    model: PainNodeClassifier,
    graph: PainGraph,
    mask: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    logits = model(graph)
    loss = torch.nn.functional.cross_entropy(logits[mask], graph.y[mask])
    accuracy = (logits[mask].argmax(dim=-1) == graph.y[mask]).float().mean()
    macro_f1 = f1_score(
        graph.y[mask].detach().cpu().numpy(),
        logits[mask].argmax(dim=-1).detach().cpu().numpy(),
        average="macro",
        zero_division=0,
    )
    return float(loss), float(accuracy), float(macro_f1)


@torch.no_grad()
def test_predictions(
    model: PainNodeClassifier,
    graph: PainGraph,
) -> dict[str, Any]:
    model.eval()
    logits = model(graph)
    indices = graph.test_mask.nonzero(as_tuple=False).view(-1)
    selected = logits[indices]
    probabilities = selected.softmax(dim=-1).cpu()
    predicted = selected.argmax(dim=-1).cpu()
    truth = graph.y[indices].cpu()
    y_true, y_pred = truth.numpy(), predicted.numpy()
    return {
        "test_node_ids": indices.cpu(),
        "y_true": truth,
        "y_pred": predicted,
        "y_prob": probabilities,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_one_run(
    config: dict[str, Any],
    *,
    variant: str,
    variant_path: str | Path,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train, select by validation Macro-F1, and evaluate test once."""
    rss_monitor = PeakRSSMonitor().start()
    set_seed(seed, bool(config["training"].get("deterministic", True)))
    device = resolve_device(str(config["device"]))
    model_config = config["model"]
    graph = load_imdb_graph(
        config["data"]["shared_path"],
        variant_path,
        reverse_paths=bool(model_config.get("reverse_paths", True)),
    )
    if int(graph.variant_meta["path_length"]) != int(model_config["path_length"]):
        raise ValueError(
            f"Artifact L={graph.variant_meta['path_length']} does not match "
            f"model L={model_config['path_length']}"
        )
    graph = graph.to(
        device, move_paths=bool(model_config.get("paths_on_device", True))
    )
    model = build_model(graph, config).to(device)

    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 20)),
        min_lr=float(training.get("min_learning_rate", 1e-5)),
    )
    epochs = int(training["epochs"])
    early_stopping = int(training.get("early_stopping_patience", epochs))
    max_hours = float(training.get("max_hours", 0.0))
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    reset_cuda_peak(device)

    best_state = None
    best_epoch = -1
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    no_improvement = 0
    history = []
    started = time.monotonic()
    time_to_best = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        train_loss = torch.nn.functional.cross_entropy(
            logits[graph.train_mask], graph.y[graph.train_mask]
        )
        train_loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        val_loss, val_accuracy, val_macro_f1 = masked_validation_metrics(
            model, graph, graph.val_mask
        )
        scheduler.step(val_loss)
        elapsed = time.monotonic() - started
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "learning_rate": current_lr,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        improved = val_macro_f1 > best_val_macro_f1 or (
            val_macro_f1 == best_val_macro_f1 and val_loss < best_val_loss
        )
        if improved:
            best_state = cpu_state_dict(model)
            best_epoch = epoch
            best_val_accuracy = val_accuracy
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            no_improvement = 0
            time_to_best = elapsed
        else:
            no_improvement += 1

        if verbose:
            print(
                f"{variant} seed={seed} epoch={epoch:04d} "
                f"train_loss={float(train_loss):.5f} "
                f"val_loss={val_loss:.5f} val_acc={val_accuracy:.5f} "
                f"val_macro_f1={val_macro_f1:.5f} "
                f"lr={current_lr:.2e}"
            )
        if no_improvement >= early_stopping:
            if verbose:
                print(f"Early stopping after {no_improvement} unimproved epochs")
            break
        if max_hours > 0 and elapsed >= max_hours * 3600:
            if verbose:
                print(f"Stopping at configured max_hours={max_hours:g}")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    training_gpu = cuda_memory_stats(device)
    model.load_state_dict(best_state)
    model.to(device)
    reset_cuda_peak(device)
    predictions = test_predictions(model, graph)
    inference_gpu = cuda_memory_stats(device)
    elapsed = time.monotonic() - started
    peak_rss = rss_monitor.stop()
    resources = {
        **model_memory_bytes(model),
        "checkpoint_bytes": serialized_torch_bytes({"model": best_state}),
        "process_peak_rss_bytes": peak_rss,
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
        "artifacts": artifact_sizes(
            {
                "shared": config["data"]["shared_path"],
                "variant": variant_path,
            }
        ),
        "environment": environment_metadata(device),
    }
    validate_resource_metrics(resources)
    return {
        "dataset": "IMDB",
        "model": "PAIN-NC",
        "variant": variant,
        "variant_source_name": graph.variant_meta.get("variant"),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "selection_metric": "validation_Macro_F1",
        "best_val_loss": best_val_loss,
        "time_to_best_seconds": time_to_best,
        "elapsed_seconds": elapsed,
        "num_nodes": graph.num_nodes,
        "num_paths": graph.num_paths,
        "history": history,
        "config": copy.deepcopy(config),
        "model_state_dict": best_state,
        "resources": resources,
        **predictions,
    }


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    validate_resource_metrics(artifact.get("resources", {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/imdb_nc.yaml")
    parser.add_argument("--variant", default="IMDb1")
    parser.add_argument("--variant-path", default="data/preprocessed/IMDB/v1_L3.pt")
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument("--device")
    parser.add_argument("--output", default="results/imdb_nc/IMDb1/seed1566911444.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    artifact = train_one_run(
        config,
        variant=args.variant,
        variant_path=args.variant_path,
        seed=args.seed,
    )
    save_artifact(artifact, args.output)
    metrics = {
        key: value
        for key, value in artifact.items()
        if key.startswith("test_") and isinstance(value, float)
    }
    print(json.dumps(metrics, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

