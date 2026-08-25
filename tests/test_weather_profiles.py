"""Regression tests for weather-specific training profiles."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from scripts.train_dual_d import apply_weather_profile, load_weather_profiles


class WeatherProfileTests(unittest.TestCase):
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

    def test_missing_profile_uses_default(self) -> None:
        profiles = {"default": {"num_workers": 8}, "profiles": {}}
        args = Namespace(num_workers=0)
        apply_weather_profile(args, "未知天气", profiles)
        self.assertEqual(args.num_workers, 8)
        self.assertEqual(args.weather_profile_name, "未知天气")


if __name__ == "__main__":
    unittest.main()
