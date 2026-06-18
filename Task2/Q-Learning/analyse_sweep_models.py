import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DT = 0.025
DEFAULT_SWEEP_ROOT = Path(__file__).resolve().parent / "top5_stab_sweep_results" / "v1"


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_error_to_upright(theta):
    return abs(wrap_to_pi(theta - math.pi))


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_telemetry(path):
    data = np.load(path, allow_pickle=True)
    if getattr(data, "shape", None) == ():
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a telemetry dictionary")

    return {
        "thetas": np.asarray(data.get("thetas", []), dtype=float),
        "omegas": np.asarray(data.get("omegas", []), dtype=float),
        "voltages": np.asarray(data.get("voltages", []), dtype=float),
        "stop_reason": str(data.get("stop_reason", "Unknown")),
    }


def max_consecutive_true(mask):
    best = 0
    current = 0
    for item in mask:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def telemetry_metrics(path):
    data = load_telemetry(path)
    thetas = data["thetas"]
    omegas = data["omegas"]
    voltages = data["voltages"]
    n_steps = min(len(thetas), len(omegas), len(voltages))

    if n_steps == 0:
        return {
            "file": path.name,
            "steps": 0,
            "duration_s": 0.0,
            "stop_reason": data["stop_reason"],
            "completed": False,
            "killed": "kill" in data["stop_reason"].lower(),
            "score": -1000.0,
        }

    thetas = thetas[:n_steps]
    omegas = omegas[:n_steps]
    voltages = voltages[:n_steps]
    errors = np.asarray([angle_error_to_upright(theta) for theta in thetas])
    abs_omegas = np.abs(omegas)

    near_upright = errors < 0.35
    controlled_upright = near_upright & (abs_omegas < 8.0)
    stable_upright = near_upright & (abs_omegas < 4.0)
    completion_text = data["stop_reason"].lower()
    completed = "completed" in completion_text and "kill" not in completion_text and "abort" not in completion_text
    killed = "kill" in completion_text or "abort" in completion_text

    duration_s = n_steps * DT
    best_error = float(np.min(errors))
    mean_best_1s_error = float(np.mean(np.sort(errors)[: min(n_steps, int(1.0 / DT))]))
    near_upright_s = float(np.sum(near_upright) * DT)
    controlled_upright_s = float(np.sum(controlled_upright) * DT)
    max_stable_s = float(max_consecutive_true(stable_upright) * DT)
    mean_abs_omega_near = float(np.mean(abs_omegas[near_upright])) if np.any(near_upright) else np.nan
    rms_voltage = float(np.sqrt(np.mean(np.square(voltages))))
    voltage_tv = float(np.sum(np.abs(np.diff(voltages)))) if n_steps > 1 else 0.0
    peak_abs_omega = float(np.max(abs_omegas))

    # Heuristic, human-readable score. It rewards getting upright, staying there
    # calmly, lasting the whole run, and avoiding violent/chattery control.
    score = 0.0
    score += 35.0 * max(0.0, 1.0 - best_error / math.pi)
    score += 25.0 * min(1.0, controlled_upright_s / 1.0)
    score += 20.0 * min(1.0, max_stable_s / 0.5)
    score += 10.0 * min(1.0, duration_s / 7.5)
    score += 10.0 if completed else 0.0
    score -= 15.0 if killed else 0.0
    score -= min(10.0, voltage_tv / 30.0)
    score -= min(10.0, max(0.0, peak_abs_omega - 38.0) / 4.0)

    return {
        "file": path.name,
        "steps": n_steps,
        "duration_s": duration_s,
        "stop_reason": data["stop_reason"],
        "completed": completed,
        "killed": killed,
        "score": float(score),
        "best_upright_error_rad": best_error,
        "mean_best_1s_error_rad": mean_best_1s_error,
        "near_upright_s": near_upright_s,
        "controlled_upright_s": controlled_upright_s,
        "max_stable_upright_s": max_stable_s,
        "mean_abs_omega_near": mean_abs_omega_near,
        "rms_voltage": rms_voltage,
        "voltage_total_variation": voltage_tv,
        "peak_abs_omega": peak_abs_omega,
    }


def classify(metrics, has_hardware):
    score = metrics.get("score", -1000.0)
    max_stable = metrics.get("max_stable_upright_s", 0.0)
    controlled = metrics.get("controlled_upright_s", 0.0)
    best_err = metrics.get("best_upright_error_rad", math.pi)
    killed = metrics.get("killed", False)

    if has_hardware and metrics.get("completed") and max_stable >= 0.5:
        return "works"
    if best_err < 0.25 and (controlled >= 0.25 or max_stable >= 0.2):
        return "almost_there"
    if best_err < 0.5 or controlled > 0.0 or score >= 40.0:
        return "promising"
    if killed:
        return "unsafe_or_unstable"
    return "not_close"


def summarize_model(model_dir):
    config = load_json(model_dir / "config.json")
    params = config.get("params", {})
    sim_file = model_dir / "last_eval_data.npy"
    hardware_files = sorted(model_dir.glob("hardware_eval*.npy"))

    sim_metrics = telemetry_metrics(sim_file) if sim_file.exists() else {}
    hardware_metrics = [telemetry_metrics(path) for path in hardware_files]
    best_hardware = max(hardware_metrics, key=lambda item: item["score"], default={})
    primary = best_hardware if best_hardware else sim_metrics

    if not config:
        return {
            "model": model_dir.name,
            "source_rank": "",
            "source_trial": "",
            "w_balance": "",
            "w_stab": "",
            "eval_score": np.nan,
            "source_eval_score": np.nan,
            "n_hardware_runs": len(hardware_metrics),
            "best_hardware_file": best_hardware.get("file", ""),
            "hardware_score": best_hardware.get("score", np.nan),
            "sim_score": sim_metrics.get("score", np.nan),
            "score_gap_hw_minus_sim": best_hardware.get("score", np.nan) - sim_metrics.get("score", np.nan),
            "classification": "incomplete",
            "best_stop_reason": "Missing config.json",
            **{f"best_{key}": value for key, value in primary.items() if key != "file"},
        }

    return {
        "model": model_dir.name,
        "source_rank": config.get("source_rank", config.get("rank", "")),
        "source_trial": config.get("source_trial_number", config.get("trial_number", "")),
        "w_balance": params.get("w_balance", params.get("w_stab", "")),
        "w_stab": params.get("w_stab", ""),
        "eval_score": safe_float(config.get("eval_score")),
        "source_eval_score": safe_float(config.get("source_eval_score")),
        "n_hardware_runs": len(hardware_metrics),
        "best_hardware_file": best_hardware.get("file", ""),
        "hardware_score": best_hardware.get("score", np.nan),
        "sim_score": sim_metrics.get("score", np.nan),
        "score_gap_hw_minus_sim": best_hardware.get("score", np.nan) - sim_metrics.get("score", np.nan),
        "classification": classify(primary, bool(best_hardware)),
        **{f"best_{key}": value for key, value in primary.items() if key != "file"},
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "classification",
        "model",
        "source_rank",
        "source_trial",
        "w_balance",
        "w_stab",
        "eval_score",
        "source_eval_score",
        "n_hardware_runs",
        "best_hardware_file",
        "hardware_score",
        "sim_score",
        "score_gap_hw_minus_sim",
        "best_duration_s",
        "best_completed",
        "best_killed",
        "best_best_upright_error_rad",
        "best_controlled_upright_s",
        "best_max_stable_upright_s",
        "best_peak_abs_omega",
        "best_voltage_total_variation",
        "best_stop_reason",
    ]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ranking_score(row):
    hardware_score = row.get("hardware_score", np.nan)
    try:
        if not np.isnan(hardware_score):
            return hardware_score
    except TypeError:
        pass
    sim_score = row.get("sim_score", np.nan)
    try:
        if not np.isnan(sim_score):
            return sim_score
    except TypeError:
        pass
    return -1000.0


def fmt(value, digits=2):
    if value == "":
        return ""
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=ranking_score, reverse=True)
    almost = [row for row in ordered if row["classification"] in {"almost_there", "promising", "works"}]

    lines = [
        "# Sweep Model Analysis",
        "",
        "The ranking below prefers hardware telemetry when present; otherwise it falls back to the saved simulation trajectory.",
        "",
        "## Almost-there candidates",
        "",
    ]

    if almost:
        lines.append("| Rank | Model | Class | HW score | Sim score | Best err rad | Controlled upright s | Stable upright s | Stop reason |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for index, row in enumerate(almost, 1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        row["model"],
                        row["classification"],
                        fmt(row.get("hardware_score")),
                        fmt(row.get("sim_score")),
                        fmt(row.get("best_best_upright_error_rad")),
                        fmt(row.get("best_controlled_upright_s")),
                        fmt(row.get("best_max_stable_upright_s")),
                        row.get("best_stop_reason", ""),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No models met the current almost-there thresholds.")

    lines.extend(
        [
            "",
            "## Full ranking",
            "",
            "| Rank | Model | Class | Eval score | HW runs | HW score | Sim score | Gap | Duration s | Peak omega | Voltage TV |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(ordered, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    row["model"],
                    row["classification"],
                    fmt(row.get("eval_score")),
                    str(row.get("n_hardware_runs", 0)),
                    fmt(row.get("hardware_score")),
                    fmt(row.get("sim_score")),
                    fmt(row.get("score_gap_hw_minus_sim")),
                    fmt(row.get("best_duration_s")),
                    fmt(row.get("best_peak_abs_omega")),
                    fmt(row.get("best_voltage_total_variation")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metric notes",
            "",
            "- `best_err_rad`: closest angular distance to upright; lower is better.",
            "- `controlled_upright_s`: time spent near upright with angular speed below 8 rad/s.",
            "- `stable_upright_s`: longest continuous near-upright segment with angular speed below 4 rad/s.",
            "- `voltage TV`: total voltage variation; high values usually mean chattering and poor transfer to hardware.",
            "- `gap`: hardware score minus simulation score; strongly negative values point to sim-to-real mismatch.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyse sweep model trajectories and identify almost-there policies.")
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT, help="Folder containing model_* directories.")
    parser.add_argument("--out-csv", type=Path, default=None, help="CSV output path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown report output path.")
    args = parser.parse_args()

    if not args.sweep_root.exists():
        raise SystemExit(f"Sweep root does not exist: {args.sweep_root}")

    model_dirs = sorted(
        path
        for path in args.sweep_root.iterdir()
        if path.is_dir() and (path.name.startswith("model_") or (path / "config.json").exists())
    )
    if not model_dirs:
        raise SystemExit(f"No model directories with config.json found in {args.sweep_root}")

    rows = [summarize_model(model_dir) for model_dir in model_dirs]
    rows.sort(key=ranking_score, reverse=True)

    out_csv = args.out_csv or args.sweep_root / "sweep_model_analysis.csv"
    out_md = args.out_md or args.sweep_root / "sweep_model_analysis.md"
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    print(f"Analysed {len(rows)} models")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote report: {out_md}")
    print()
    print("Top candidates:")
    for row in rows[:5]:
        print(
            f"- {row['classification']:>16} | {row['model']} | "
            f"hw={fmt(row.get('hardware_score'))} sim={fmt(row.get('sim_score'))} | "
            f"stable={fmt(row.get('best_max_stable_upright_s'))}s | "
            f"err={fmt(row.get('best_best_upright_error_rad'))}rad"
        )


if __name__ == "__main__":
    main()
