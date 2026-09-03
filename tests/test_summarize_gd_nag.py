import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_gd_nag


FIELDNAMES = [
    "dataset", "method", "epoch", "homomorphic_seconds", "refresh_seconds",
    "metric_decryption_seconds", "paired_simulated_refresh_seconds",
    "seconds_per_epoch", "accuracy", "loss", "max_plaintext_model_error",
    "refreshed", "level_before_refresh", "level_after_refresh", "optimizer",
    "momentum", "nag_packing",
]


class SummarizeGdNagTests(unittest.TestCase):
    def _write_run(self, directory, run, optimizer, losses, seconds):
        path = directory / f"run_{run:03d}_{optimizer}.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            for index, (loss, epoch_seconds) in enumerate(zip(losses, seconds), start=1):
                writer.writerow({
                    "dataset": "fixture", "method": "simulated_bootstrapping",
                    "epoch": index, "homomorphic_seconds": epoch_seconds - 0.1,
                    "refresh_seconds": 0.1, "metric_decryption_seconds": 0,
                    "paired_simulated_refresh_seconds": 0,
                    "seconds_per_epoch": epoch_seconds, "accuracy": 0.5 + index / 10,
                    "loss": loss, "max_plaintext_model_error": 1e-8 * index,
                    "refreshed": 1, "level_before_refresh": 10,
                    "level_after_refresh": 0, "optimizer": optimizer,
                    "momentum": 0.1 if optimizer == "nag" else 0,
                    "nag_packing": "packed" if optimizer == "nag" else "separate",
                })
        return path

    def test_reports_fixed_epoch_and_target_convergence_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root, raw = Path(temporary_directory), Path(temporary_directory) / "raw"
            raw.mkdir()
            self._write_run(raw, 1, "gd", [0.7, 0.6, 0.5], [1.0, 1.0, 1.0])
            self._write_run(raw, 1, "nag", [0.7, 0.49, 0.4], [1.2, 1.2, 1.2])
            metric_rows, comparison_rows = summarize_gd_nag.summarize_directory(
                raw, root / "summary"
            )
            self.assertEqual((metric_rows, comparison_rows), (2, 1))
            with (root / "summary/per_run_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                comparison = next(csv.DictReader(source))
            self.assertAlmostEqual(float(comparison["nag_final_loss_improvement"]), 0.1)
            self.assertEqual(comparison["nag_packing"], "packed")
            self.assertEqual(int(comparison["nag_epochs_to_gd_final_loss"]), 2)
            self.assertEqual(int(comparison["epoch_savings_at_gd_final_loss"]), 1)
            self.assertAlmostEqual(float(comparison["target_time_speedup_gd_over_nag"]), 1.25)
            self.assertAlmostEqual(
                float(comparison["fixed_epoch_runtime_ratio_gd_over_nag"]), 3.0 / 3.6
            )
            with (root / "summary/aggregate_epoch_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                epoch_rows = list(csv.DictReader(source))
            nag_epoch_two = next(
                row for row in epoch_rows
                if row["optimizer"] == "nag" and row["epoch"] == "2"
            )
            self.assertAlmostEqual(float(nag_epoch_two["loss_mean"]), 0.49)
            self.assertAlmostEqual(float(nag_epoch_two["cumulative_seconds_mean"]), 2.4)

    def test_rejects_filename_optimizer_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root, raw = Path(temporary_directory), Path(temporary_directory) / "raw"
            raw.mkdir()
            path = self._write_run(raw, 1, "nag", [0.7], [1.0])
            path.rename(raw / "run_001_gd.csv")
            with self.assertRaisesRegex(ValueError, "filename says"):
                summarize_gd_nag.summarize_directory(raw, root / "summary")


if __name__ == "__main__":
    unittest.main()
