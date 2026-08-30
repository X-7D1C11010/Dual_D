"""Regression tests for weather-specific training profiles."""

from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.train_dual_d import apply_weather_profile, load_weather_profiles


class WeatherProfileTests(unittest.TestCase):
    def test_v12_profiles_strengthen_the_requested_training_controls(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        v11 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v11.json"
        )["profiles"]
        fog_v5 = load_weather_profiles(
            project_root / "configs" / "module_c_fog_v5_frozen.json"
        )["profiles"]["雾天"]
        v12 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v12.json"
        )["profiles"]

        baselines = {
            "黑天": v11["黑天"],
            "逆光": v11["逆光"],
            "雾天": fog_v5,
            "雨天": v11["雨天"],
        }
        self.assertEqual(set(v12), set(baselines))
        for domain, baseline in baselines.items():
            candidate = v12[domain]
            baseline_vis = baseline.get(
                "vis_augmentation_strength",
                baseline["augmentation_strength"],
            )
            baseline_ir = baseline.get(
                "ir_augmentation_strength",
                baseline["augmentation_strength"],
            )
            self.assertGreater(
                candidate["augmentation_strength"],
                baseline["augmentation_strength"],
            )
            self.assertGreater(candidate["vis_augmentation_strength"], baseline_vis)
            self.assertGreater(candidate["ir_augmentation_strength"], baseline_ir)
            for key in ("label_smoothing", "classifier_dropout", "weight_decay"):
                self.assertGreater(candidate[key], baseline[key])
            for key in ("lr_main", "lr_visual", "lr_discriminator"):
                self.assertLess(candidate[key], baseline[key])
            for key in (
                "adversarial_warmup_epochs",
                "adversarial_ramp_epochs",
                "module_c_warmup_epochs",
                "module_c_ramp_epochs",
            ):
                self.assertGreater(candidate[key], baseline[key])
            self.assertGreater(candidate["checkpoint_selection_min_epoch"], 1)
            self.assertGreaterEqual(
                candidate["early_stopping_min_epochs"],
                candidate["checkpoint_selection_min_epoch"],
            )
            self.assertEqual(
                candidate["dual_loss_weights"],
                baseline["dual_loss_weights"],
            )

    def test_v11_profiles_lower_only_target_classification_weight(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        v10 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v10.json"
        )
        v11 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v11.json"
        )
        expected = deepcopy(v10["profiles"])
        expected["黑天"]["target_classification_weight"] = 0.80
        expected["逆光"]["target_classification_weight"] = 0.70
        expected["雨天"]["target_classification_weight"] = 0.85
        self.assertEqual(v11["profiles"], expected)

    def test_v10_profiles_change_only_the_preregistered_single_factors(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        v9 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v9.json"
        )
        v10 = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v10.json"
        )
        expected = deepcopy(v9["profiles"])
        expected["黑天"]["vis_augmentation_strength"] = 0.50
        expected["黑天"]["ir_augmentation_strength"] = 0.50
        expected["逆光"]["dual_loss_weights"]["contrastive"] = 0.020
        expected["雨天"]["dual_loss_weights"]["prototype_contrastive"] = 0.015
        self.assertEqual(v10["profiles"], expected)

    def test_v9_profiles_match_target_epoch_sizes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        profiles = load_weather_profiles(
            project_root / "configs" / "module_c_weather_profiles_v9.json"
        )
        expected = {
            "黑天": (16, 30, 0.55, 0.35),
            "逆光": (16, 9, 0.70, 0.40),
            "雨天": (16, 6, 0.60, 0.40),
        }
        for domain, (batch_size, steps, visible, infrared) in expected.items():
            args = Namespace(dual_loss_weights={})
            apply_weather_profile(args, domain, profiles)
            self.assertEqual(args.batch_size, batch_size)
            self.assertEqual(args.min_steps_per_epoch, steps)
            self.assertEqual(args.vis_augmentation_strength, visible)
            self.assertEqual(args.ir_augmentation_strength, infrared)
            self.assertTrue(args.freeze_frozen_batch_norm_stats)
            self.assertEqual(args.monitor_stability_window, 3)

    def test_separate_profile_files_are_loaded_relative_to_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "rain.json").write_text(
                json.dumps(
                    {
                        "min_steps_per_epoch": 4,
                        "lr_scheduler_start_epoch": 24,
                    }
                ),
                encoding="utf-8",
            )
            index = root / "profiles.json"
            index.write_text(
                json.dumps({"profile_files": {"雨天": "rain.json"}}),
                encoding="utf-8",
            )

            profiles = load_weather_profiles(index)
            self.assertEqual(profiles["profiles"]["雨天"]["min_steps_per_epoch"], 4)
            self.assertEqual(
                profiles["profiles"]["雨天"]["lr_scheduler_start_epoch"], 24
            )

    def test_profile_overrides_are_applied_without_mutating_source_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "default": {"num_workers": 4},
                        "profiles": {
                            "雨天": {"batch_size": 32, "min_steps_per_epoch": 8},
                        },
                    }
                ),
                encoding="utf-8",
            )
            profiles = load_weather_profiles(path)
            source = Namespace(batch_size=64, num_workers=0, min_steps_per_epoch=4)
            run_args = Namespace(**vars(source))
            overrides = apply_weather_profile(run_args, "雨天", profiles)
            self.assertEqual(vars(source), {"batch_size": 64, "num_workers": 0, "min_steps_per_epoch": 4})
            self.assertEqual(run_args.batch_size, 32)
            self.assertEqual(run_args.num_workers, 4)
            self.assertEqual(run_args.min_steps_per_epoch, 8)
            self.assertEqual(run_args.weather_profile_name, "雨天")
            self.assertEqual(overrides["batch_size"], 32)

    def test_unknown_profile_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profiles.json"
            path.write_text(json.dumps({"雨天": {"epochs": 20}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_weather_profiles(path)

    def test_dual_loss_weights_are_validated_and_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "雨天": {
                            "monitor_stability_window": 5,
                            "vis_augmentation_strength": 0.75,
                            "ir_augmentation_strength": 0.40,
                            "freeze_frozen_batch_norm_stats": True,
                            "dual_loss_weights": {
                                "classification": 0.65,
                                "cycle": 0.28,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            profiles = load_weather_profiles(path)
            args = Namespace(monitor_stability_window=1, dual_loss_weights={})
            overrides = apply_weather_profile(args, "雨天", profiles)
            self.assertEqual(args.monitor_stability_window, 5)
            self.assertEqual(args.dual_loss_weights["classification"], 0.65)
            self.assertEqual(overrides["dual_loss_weights"]["cycle"], 0.28)
            self.assertEqual(args.vis_augmentation_strength, 0.75)
            self.assertEqual(args.ir_augmentation_strength, 0.40)
            self.assertTrue(args.freeze_frozen_batch_norm_stats)

            path.write_text(
                json.dumps({"雨天": {"dual_loss_weights": {"unknown": 0.1}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_weather_profiles(path)

    def test_missing_profile_uses_default(self) -> None:
        profiles = {"default": {"num_workers": 8}, "profiles": {}}
        args = Namespace(num_workers=0)
        apply_weather_profile(args, "未知天气", profiles)
        self.assertEqual(args.num_workers, 8)
        self.assertEqual(args.weather_profile_name, "未知天气")


if __name__ == "__main__":
    unittest.main()
