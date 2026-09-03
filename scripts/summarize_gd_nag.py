#!/usr/bin/env python3
"""Summarize repeated, paired GD and NAG benchmark CSV files."""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILE_PATTERN = re.compile(r"^run_(?P<run>0*[1-9][0-9]*)_(?P<optimizer>gd|nag)\.csv$")
REQUIRED_COLUMNS = {
    "dataset", "method", "epoch", "homomorphic_seconds", "refresh_seconds",
    "metric_decryption_seconds", "paired_simulated_refresh_seconds",
    "seconds_per_epoch", "accuracy", "loss", "max_plaintext_model_error",
    "refreshed", "level_before_refresh", "level_after_refresh", "optimizer",
    "momentum",
}
OPTIMIZER_METRICS = [
    "final_loss", "minimum_loss", "mean_epoch_loss", "final_accuracy",
    "total_seconds", "mean_seconds_per_epoch", "total_homomorphic_seconds",
    "total_refresh_seconds", "total_metric_decryption_seconds",
    "total_paired_simulated_refresh_seconds", "refresh_count",
    "max_plaintext_model_error", "max_level_before_refresh",
    "final_level_after_refresh",
]
COMPARISON_METRICS = [
    "gd_final_loss", "nag_final_loss", "nag_final_loss_improvement",
    "gd_final_accuracy", "nag_final_accuracy", "nag_final_accuracy_improvement",
    "gd_total_seconds", "nag_total_seconds",
    "fixed_epoch_runtime_ratio_gd_over_nag",
]
TARGET_METRICS = [
    "nag_epochs_to_gd_final_loss", "nag_seconds_to_gd_final_loss",
    "epoch_savings_at_gd_final_loss", "target_time_speedup_gd_over_nag",
]


def _number(row, column, source):
    try:
        value = float(row[column])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{source}: invalid {column!r} value") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}: {column!r} must be finite")
    return value


def _integer(row, column, source):
    value = _number(row, column, source)
    if not value.is_integer():
        raise ValueError(f"{source}: {column!r} must be an integer")
    return int(value)


def _mean_stddev(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_run_file(path, run, expected_optimizer):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{path}: benchmark CSV is empty")

    grouped = defaultdict(list)
    for row in raw_rows:
        optimizer = row["optimizer"].strip().lower()
        if optimizer != expected_optimizer:
            raise ValueError(
                f"{path}: filename says {expected_optimizer!r}, but a row says {optimizer!r}"
            )
        grouped[(row["dataset"].strip(), row["method"].strip())].append(row)

    summaries = []
    for (dataset, method), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: _integer(row, "epoch", path))
        epochs = [_integer(row, "epoch", path) for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError(f"{path}: {dataset}/{method} epochs must start at 1 and be contiguous")
        momenta = {_number(row, "momentum", path) for row in rows}
        if len(momenta) != 1:
            raise ValueError(f"{path}: {dataset}/{method} uses multiple momentum values")
        momentum = momenta.pop()
        if expected_optimizer == "gd" and momentum != 0.0:
            raise ValueError(f"{path}: GD rows must report effective momentum 0")
        # Pre-state-packing benchmark files used the separate layout implicitly.
        nag_packings = {
            row.get("nag_packing", row.get("nag_state_packing", "separate")).strip().lower()
            for row in rows
        }
        if len(nag_packings) != 1:
            raise ValueError(f"{path}: {dataset}/{method} uses multiple NAG packing modes")
        nag_packing = nag_packings.pop()
        if nag_packing not in {"separate", "packed"}:
            raise ValueError(f"{path}: unknown NAG packing {nag_packing!r}")
        if expected_optimizer == "gd" and nag_packing != "separate":
            raise ValueError(f"{path}: GD rows must report separate NAG packing")

        losses = [_number(row, "loss", path) for row in rows]
        accuracies = [_number(row, "accuracy", path) for row in rows]
        epoch_seconds = [_number(row, "seconds_per_epoch", path) for row in rows]
        minimum_loss = min(losses)
        summaries.append({
            "run": run,
            "dataset": dataset,
            "method": method,
            "optimizer": expected_optimizer,
            "momentum": momentum,
            "nag_packing": nag_packing,
            "epochs": len(rows),
            "final_loss": losses[-1],
            "minimum_loss": minimum_loss,
            "minimum_loss_epoch": losses.index(minimum_loss) + 1,
            "mean_epoch_loss": statistics.mean(losses),
            "final_accuracy": accuracies[-1],
            "total_seconds": sum(epoch_seconds),
            "mean_seconds_per_epoch": statistics.mean(epoch_seconds),
            "total_homomorphic_seconds": sum(_number(row, "homomorphic_seconds", path) for row in rows),
            "total_refresh_seconds": sum(_number(row, "refresh_seconds", path) for row in rows),
            "total_metric_decryption_seconds": sum(
                _number(row, "metric_decryption_seconds", path) for row in rows
            ),
            "total_paired_simulated_refresh_seconds": sum(
                _number(row, "paired_simulated_refresh_seconds", path) for row in rows
            ),
            "refresh_count": sum(_integer(row, "refreshed", path) for row in rows),
            "max_plaintext_model_error": max(
                _number(row, "max_plaintext_model_error", path) for row in rows
            ),
            "max_level_before_refresh": max(
                _integer(row, "level_before_refresh", path) for row in rows
            ),
            "final_level_after_refresh": _integer(rows[-1], "level_after_refresh", path),
            "_losses": losses,
            "_accuracies": accuracies,
            "_epoch_seconds": epoch_seconds,
        })
    return summaries


def _build_comparisons(per_run):
    pairs = defaultdict(dict)
    for row in per_run:
        key = (row["run"], row["dataset"], row["method"])
        if row["optimizer"] in pairs[key]:
            raise ValueError(f"Duplicate optimizer result for run/dataset/method {key}")
        pairs[key][row["optimizer"]] = row

    comparisons = []
    for (run, dataset, method), pair in sorted(pairs.items()):
        if set(pair) != {"gd", "nag"}:
            raise ValueError(f"Missing paired GD or NAG result for run {run}, {dataset}/{method}")
        gd, nag = pair["gd"], pair["nag"]
        if gd["epochs"] != nag["epochs"]:
            raise ValueError(f"GD and NAG epoch counts differ for run {run}, {dataset}/{method}")
        target_loss = gd["final_loss"]
        reached_epoch = next(
            (index + 1 for index, loss in enumerate(nag["_losses"]) if loss <= target_loss),
            None,
        )
        reached_seconds = (
            sum(nag["_epoch_seconds"][:reached_epoch]) if reached_epoch is not None else None
        )
        comparisons.append({
            "run": run,
            "dataset": dataset,
            "method": method,
            "momentum": nag["momentum"],
            "nag_packing": nag["nag_packing"],
            "epochs": gd["epochs"],
            "gd_final_loss": target_loss,
            "nag_final_loss": nag["final_loss"],
            "nag_final_loss_improvement": target_loss - nag["final_loss"],
            "gd_final_accuracy": gd["final_accuracy"],
            "nag_final_accuracy": nag["final_accuracy"],
            "nag_final_accuracy_improvement": nag["final_accuracy"] - gd["final_accuracy"],
            "gd_total_seconds": gd["total_seconds"],
            "nag_total_seconds": nag["total_seconds"],
            "fixed_epoch_runtime_ratio_gd_over_nag": (
                gd["total_seconds"] / nag["total_seconds"] if nag["total_seconds"] else ""
            ),
            "gd_final_loss_target": target_loss,
            "nag_reached_gd_final_loss": int(reached_epoch is not None),
            "nag_epochs_to_gd_final_loss": reached_epoch if reached_epoch is not None else "",
            "nag_seconds_to_gd_final_loss": reached_seconds if reached_seconds is not None else "",
            "epoch_savings_at_gd_final_loss": (
                gd["epochs"] - reached_epoch if reached_epoch is not None else ""
            ),
            "target_time_speedup_gd_over_nag": (
                gd["total_seconds"] / reached_seconds if reached_seconds else ""
            ),
        })
    return comparisons


def _aggregate(rows, key_names, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in key_names)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(key_names, key))
        summary["runs"] = len(group)
        for metric in metrics:
            mean, stddev = _mean_stddev([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_stddev"] = stddev
        output.append(summary)
    return output


def _aggregate_comparisons(comparisons):
    groups = defaultdict(list)
    for row in comparisons:
        groups[(
            row["dataset"], row["method"], row["momentum"],
            row["nag_packing"], row["epochs"],
        )].append(row)
    output = []
    for (dataset, method, momentum, nag_packing, epochs), rows in sorted(groups.items()):
        summary = {
            "dataset": dataset, "method": method, "momentum": momentum,
            "nag_packing": nag_packing, "epochs": epochs, "runs": len(rows),
            "nag_target_success_rate": statistics.mean(
                row["nag_reached_gd_final_loss"] for row in rows
            ),
        }
        for metric in COMPARISON_METRICS:
            mean, stddev = _mean_stddev([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_stddev"] = stddev
        successes = [row for row in rows if row["nag_reached_gd_final_loss"]]
        for metric in TARGET_METRICS:
            if successes:
                mean, stddev = _mean_stddev([float(row[metric]) for row in successes])
                summary[f"{metric}_mean"], summary[f"{metric}_stddev"] = mean, stddev
            else:
                summary[f"{metric}_mean"], summary[f"{metric}_stddev"] = "", ""
        output.append(summary)
    return output


def summarize_directory(input_directory, output_directory):
    input_files = [
        path for path in sorted(input_directory.glob("run_*_*.csv"))
        if path.is_file() and FILE_PATTERN.match(path.name)
    ]
    if not input_files:
        raise ValueError(f"No paired run_NNN_gd.csv/run_NNN_nag.csv files found in {input_directory}")
    per_run = []
    for path in input_files:
        match = FILE_PATTERN.match(path.name)
        per_run.extend(_read_run_file(path, int(match.group("run")), match.group("optimizer")))
    comparisons = _build_comparisons(per_run)
    optimizer_aggregates = _aggregate(
        per_run,
        ["dataset", "method", "optimizer", "momentum", "nag_packing", "epochs"],
        OPTIMIZER_METRICS,
    )
    comparison_aggregates = _aggregate_comparisons(comparisons)
    epoch_rows = []
    for row in per_run:
        cumulative_seconds = 0.0
        for epoch, (loss, accuracy, seconds) in enumerate(
            zip(row["_losses"], row["_accuracies"], row["_epoch_seconds"]), start=1
        ):
            cumulative_seconds += seconds
            epoch_rows.append({
                "run": row["run"], "dataset": row["dataset"], "method": row["method"],
                "optimizer": row["optimizer"], "momentum": row["momentum"],
                "nag_packing": row["nag_packing"],
                "epoch": epoch, "loss": loss, "accuracy": accuracy,
                "seconds_per_epoch": seconds, "cumulative_seconds": cumulative_seconds,
            })
    epoch_aggregates = _aggregate(
        epoch_rows,
        ["dataset", "method", "optimizer", "momentum", "nag_packing", "epoch"],
        ["loss", "accuracy", "seconds_per_epoch", "cumulative_seconds"],
    )

    visible_per_run = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in per_run
    ]
    _write_csv(output_directory / "per_run_metrics.csv", visible_per_run)
    _write_csv(output_directory / "per_run_comparison.csv", comparisons)
    _write_csv(output_directory / "aggregate_optimizer_metrics.csv", optimizer_aggregates)
    _write_csv(output_directory / "aggregate_epoch_metrics.csv", epoch_aggregates)
    _write_csv(output_directory / "aggregate_comparison.csv", comparison_aggregates)

    for row in comparison_aggregates:
        target = row["nag_epochs_to_gd_final_loss_mean"]
        target_text = f"{target:.2f} epochs" if target != "" else "not reached"
        print(
            f"{row['dataset']}/{row['method']}: "
            f"NAG packing {row['nag_packing']}; "
            f"loss GD {row['gd_final_loss_mean']:.6g}, NAG {row['nag_final_loss_mean']:.6g}; "
            f"time GD {row['gd_total_seconds_mean']:.3f}s, "
            f"NAG {row['nag_total_seconds_mean']:.3f}s; "
            f"NAG reaches GD final loss: {target_text}"
        )
    return len(per_run), len(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        metric_rows, comparison_rows = summarize_directory(
            arguments.input_dir, arguments.output_dir
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Wrote {metric_rows} per-optimizer rows and {comparison_rows} paired rows "
        f"to {arguments.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
