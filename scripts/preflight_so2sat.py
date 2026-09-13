#!/usr/bin/env python3
"""Run one real So2Sat batch through the complete Satellite-Dual_D graph.

This preflight performs forward and backward diagnostics without taking an
optimizer step, writing a checkpoint, or reading the official testing split's
samples.  It is intended to run before any smoke or full training job.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.data import PairedClassSampler  # noqa: E402
from dual_d.models import LabelSmoothingCrossEntropy  # noqa: E402
from dual_d.training.trainer import (  # noqa: E402
    _audit_so2sat_protocol,
    _encode_batch_modalities,
    build_datasets,
    build_models,
    resolve_device,
    validate_cuda_architecture,
)
from scripts.train_dual_d import build_parser, load_json_defaults  # noqa: E402


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).cpu())
        squared += value * value
    return squared**0.5


def _module_gradient_report(models) -> dict[str, dict[str, float | int]]:
    modules = {
        "sar_encoder": models.net_ir,
        "optical_encoder": models.net_vis,
        "tensor_alignment": models.tal,
        "translator": models.dual_adapter.coordinator.translator,
        "classifier": models.classifier,
    }
    report = {}
    for name, module in modules.items():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        report[name] = {
            "trainable_parameter_count": int(sum(p.numel() for p in parameters)),
            "parameters_with_gradient": int(sum(p.grad is not None for p in parameters)),
            "gradient_norm": _gradient_norm(parameters),
        }
    return report


def run_preflight(args) -> dict[str, object]:
    """Return a JSON-serializable one-batch diagnostic report."""

    if getattr(args, "dataset_type", "") != "so2sat_lcz42":
        raise ValueError("The preflight requires dataset_type=so2sat_lcz42.")
    device = resolve_device(args.device)
    validate_cuda_architecture(device)
    (
        source_train,
        _source_eval,
        target_train,
        _target_train_eval,
        target_adapt_val,
        target_test,
        label_map,
    ) = build_datasets(args)
    datasets = (
        source_train,
        _source_eval,
        target_train,
        _target_train_eval,
        target_adapt_val,
        target_test,
    )
    try:
        audit = _audit_so2sat_protocol(
            source_train,
            target_train,
            target_adapt_val,
            target_test,
        )
        if audit["leakage_detected"] or not audit["closed_set_17_classes"]:
            raise RuntimeError(f"So2Sat protocol audit failed: {audit}")

        paired_loader = PairedClassSampler(
            source_train,
            target_train,
            batch_size=int(args.batch_size),
            min_steps_per_epoch=1,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )
        source_batch, target_batch = next(iter(paired_loader))
        source_labels = source_batch["label"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)

        models = build_models(args, num_classes=len(label_map), device=device)
        for module in models.__dict__.values():
            if module is not None:
                module.train()
                module.zero_grad(set_to_none=True)

        source_modalities = _encode_batch_modalities(models, source_batch, device)
        target_modalities = _encode_batch_modalities(models, target_batch, device)
        projected_source, projected_target, tensor_loss = models.tal(
            source_modalities,
            target_modalities,
        )
        source_features = torch.cat(projected_source, dim=1)
        target_features = torch.cat(projected_target, dim=1)
        outputs = models.dual_adapter.forward_features(source_features, target_features)

        discriminator_loss, discriminator_logs = (
            models.dual_adapter.compute_discriminator_loss(outputs)
        )
        if not bool(torch.isfinite(discriminator_loss)):
            raise FloatingPointError("Non-finite discriminator loss in preflight.")
        discriminator_loss.backward()
        discriminator_gradient_norm = _gradient_norm(
            models.dual_adapter.discriminator_parameters()
        )
        if not math.isfinite(discriminator_gradient_norm) or discriminator_gradient_norm <= 0.0:
            raise RuntimeError(
                "Discriminator gradients are zero or non-finite in preflight: "
                f"{discriminator_gradient_norm}"
            )
        models.dual_adapter.zero_grad(set_to_none=True)
        models.dual_adapter.set_discriminators_trainable(False)

        criterion = LabelSmoothingCrossEntropy(eps=float(args.label_smoothing))
        source_classification = criterion(
            models.classifier(source_features), source_labels
        )
        target_classification = criterion(
            models.classifier(target_features), target_labels
        )
        classification_loss = source_classification + float(
            args.target_classification_weight
        ) * target_classification
        generator_loss, generator_logs = models.dual_adapter.compute_generator_loss(
            outputs=outputs,
            classifier=models.classifier,
            criterion_cls=criterion,
            source_labels=source_labels,
            target_labels=target_labels,
            num_classes=len(label_map),
            adversarial_scale=1.0,
            module_c_scale=1.0,
            modality_drift_scale=1.0,
        )
        total_loss = classification_loss + float(args.tal_weight) * tensor_loss + generator_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("Non-finite main loss in preflight.")
        total_loss.backward()
        gradient_report = _module_gradient_report(models)
        missing_gradients = [
            name
            for name, values in gradient_report.items()
            if int(values["parameters_with_gradient"]) == 0
            or not math.isfinite(float(values["gradient_norm"]))
            or float(values["gradient_norm"]) <= 0.0
        ]
        if missing_gradients:
            raise RuntimeError(
                "No gradients reached required modules: " + ", ".join(missing_gradients)
            )

        report = {
            "status": "passed",
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "compiled_cuda_arches": (
                torch.cuda.get_arch_list() if torch.cuda.is_available() else []
            ),
            "protocol_audit": audit,
            "shape_trace": {
                "source_sar": list(source_batch["sar"].shape),
                "source_optical": list(source_batch["optical"].shape),
                "target_sar": list(target_batch["sar"].shape),
                "target_optical": list(target_batch["optical"].shape),
                "source_encoder_features": [list(value.shape) for value in source_modalities],
                "target_encoder_features": [list(value.shape) for value in target_modalities],
                "source_projected_features": [list(value.shape) for value in projected_source],
                "target_projected_features": [list(value.shape) for value in projected_target],
                "source_fused": list(source_features.shape),
                "target_fused": list(target_features.shape),
                "source_to_target": list(outputs.target_like.shape),
                "target_to_source": list(outputs.source_like.shape),
            },
            "losses": {
                "classification_source": float(source_classification.detach().cpu()),
                "classification_target": float(target_classification.detach().cpu()),
                "tensor_alignment": float(tensor_loss.detach().cpu()),
                "discriminator": float(discriminator_loss.detach().cpu()),
                "generator": float(generator_loss.detach().cpu()),
                "total_main": float(total_loss.detach().cpu()),
            },
            "discriminator_gradient_norm": discriminator_gradient_norm,
            "main_gradients": gradient_report,
            "discriminator_logs": discriminator_logs,
            "generator_logs": generator_logs,
        }
        return report
    finally:
        for dataset in datasets:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "so2sat_lcz42.json"),
    )
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    cli = parser.parse_args()

    config_path = Path(cli.config)
    defaults = load_json_defaults(config_path)
    args = build_parser(defaults).parse_args([])
    args.dataset_root = cli.dataset_root or args.dataset_root
    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.device = cli.device
    dual_config = Path(args.dual_config)
    if not dual_config.is_absolute():
        args.dual_config = str(PROJECT_ROOT / dual_config)

    report = run_preflight(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if cli.output:
        output = Path(cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
