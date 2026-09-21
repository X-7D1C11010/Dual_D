"""Fail-fast M4-SAR integration preflight without an epoch loop or test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dual_d import build_parser, load_json_defaults  # noqa: E402
from dual_d.training.trainer import (  # noqa: E402
    IndependentDomainLoaders,
    build_classification_criterion,
    build_datasets,
    build_models,
    build_optimizers,
    extract_fused_features,
    resolve_device,
    set_seed,
    validate_cuda_architecture,
)


def preflight(args) -> dict[str, object]:
    """Run one disposable forward/backward optimizer smoke step on train only."""

    if args.dataset_type != "m4sar_classification":
        raise ValueError("M4-SAR preflight requires dataset_type=m4sar_classification.")
    if args.model_mode != "dual_d":
        raise ValueError("This preflight checks the full model_mode=dual_d path.")
    args.evaluate_target_test = False
    args.save_checkpoints = False
    args.save_feature_embeddings = False
    args.multi_gpu = False
    set_seed(int(args.seed), deterministic=False)
    device = resolve_device(args.device)
    validate_cuda_architecture(device)

    (
        source_train,
        _source_eval,
        target_train,
        _target_eval,
        target_val,
        target_test,
        label_map,
    ) = build_datasets(args)
    if target_test is not None:
        raise AssertionError("Preflight must not construct M4-SAR Target test.")
    args.pin_memory = device.type == "cuda"
    independent_loaders = IndependentDomainLoaders(source_train, target_train, args)
    if len(independent_loaders) < 1:
        raise ValueError("Training split is smaller than preflight batch_size.")
    source_batch, target_batch = next(iter(independent_loaders))
    source_pair_ids = list(source_batch["pair_id"])
    target_pair_ids = list(target_batch["pair_id"])
    if source_pair_ids == target_pair_ids:
        raise AssertionError(
            "Independent Source/Target shuffles produced identical first-batch order; "
            "change the seed and rerun the preflight."
        )
    expected_size = int(args.m4sar_input_size)
    if tuple(source_batch["optical"].shape[1:]) != (3, expected_size, expected_size):
        raise AssertionError("Source Optical tensor is not [N,3,H,W].")
    if tuple(source_batch["sar"].shape[1:]) != (1, expected_size, expected_size):
        raise AssertionError("Source SAR tensor is not [N,1,H,W].")
    if tuple(target_batch["optical"].shape[1:]) != (3, expected_size, expected_size):
        raise AssertionError("Target Optical tensor is not [N,3,H,W].")
    if tuple(target_batch["sar"].shape[1:]) != (1, expected_size, expected_size):
        raise AssertionError("Target SAR tensor is not [N,1,H,W].")
    for name, labels in (
        ("Source", source_batch["label"]),
        ("Target", target_batch["label"]),
    ):
        if int(labels.min()) < 0 or int(labels.max()) >= 6:
            raise AssertionError(f"{name} labels are outside [0,5].")

    models = build_models(args, len(label_map), device)
    criterion = build_classification_criterion(
        args,
        source_train,
        device,
        len(label_map),
    )
    optimizer_main, optimizer_disc = build_optimizers(args, models)
    source_labels = source_batch["label"].to(device)
    target_labels = target_batch["label"].to(device)
    feat_src, feat_tgt, loss_tal = extract_fused_features(
        models,
        source_batch,
        target_batch,
        device,
        args,
    )
    outputs = models.dual_adapter.forward_features(feat_src, feat_tgt)
    if not bool(torch.isfinite(feat_src).all()) or not bool(torch.isfinite(feat_tgt).all()):
        raise FloatingPointError("M4-SAR preflight produced non-finite fused features.")

    def gradient_check(parameters) -> dict[str, object]:
        gradients = [
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None
        ]
        finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
        nonzero = bool(gradients) and any(bool(torch.count_nonzero(grad)) for grad in gradients)
        return {"present": bool(gradients), "finite": finite, "nonzero": nonzero}

    optimizer_disc.zero_grad(set_to_none=True)
    loss_disc, _ = models.dual_adapter.compute_discriminator_loss(outputs)
    loss_disc.backward()
    discriminator_gradients = gradient_check(
        models.dual_adapter.discriminator_parameters()
    )
    optimizer_disc.step()

    models.dual_adapter.set_discriminators_trainable(False)
    optimizer_main.zero_grad(set_to_none=True)
    loss_cls = criterion(models.classifier(feat_src), source_labels)
    loss_cls = loss_cls + float(args.target_classification_weight) * criterion(
        models.classifier(feat_tgt),
        target_labels,
    )
    loss_dual, _ = models.dual_adapter.compute_generator_loss(
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
    loss_main = loss_cls + float(args.tal_weight) * loss_tal + loss_dual
    loss_main.backward()
    gradient_checks = {
        "optical_encoder": gradient_check(models.net_vis.parameters()),
        "sar_encoder": gradient_check(models.net_ir.parameters()),
        "tensor_alignment": gradient_check(models.tal.parameters()),
        "translator": gradient_check(models.dual_adapter.generator_parameters()),
        "classifier": gradient_check(models.classifier.parameters()),
        "discriminators": discriminator_gradients,
    }
    failed_gradients = [
        name
        for name, status in gradient_checks.items()
        if not (status["present"] and status["finite"] and status["nonzero"])
    ]
    if failed_gradients:
        raise AssertionError(
            "Missing, non-finite, or zero required gradients: "
            + ", ".join(failed_gradients)
        )
    optimizer_main.step()
    if int(getattr(models.tal, "orthogonalize_interval", 1)) > 0:
        models.tal.apply_orthogonal_projection()
    models.dual_adapter.set_discriminators_trainable(True)
    if not bool(torch.isfinite(loss_main)) or not bool(torch.isfinite(loss_disc)):
        raise FloatingPointError("M4-SAR preflight produced a non-finite loss.")

    return {
        "status": "ok",
        "device": str(device),
        "source_train_samples": len(source_train),
        "target_train_samples": len(target_train),
        "target_val_samples": len(target_val),
        "target_test_opened": False,
        "source_target_loader_policy": "independent_shuffle_no_pair_correspondence",
        "source_batch_pair_ids": source_pair_ids,
        "target_batch_pair_ids": target_pair_ids,
        "optical_shape": list(source_batch["optical"].shape),
        "sar_shape": list(source_batch["sar"].shape),
        "fused_shape": list(feat_src.shape),
        "loss_main": float(loss_main.detach().cpu()),
        "loss_discriminator": float(loss_disc.detach().cpu()),
        "required_gradients": gradient_checks,
        "class_weights": args.effective_class_weights,
        "note": "Disposable smoke step only; no epoch loop, checkpoint, or Target test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "m4sar_classification.json"),
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    cli = parser.parse_args()

    defaults = load_json_defaults(cli.config)
    args = build_parser(defaults).parse_args([])
    if cli.manifest:
        args.m4sar_manifest = cli.manifest
    if cli.data_root:
        args.m4sar_data_root = cli.data_root
    args.device = cli.device
    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.persistent_workers = False
    print(json.dumps(preflight(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
