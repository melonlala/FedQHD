#!/usr/bin/env python3
"""Rerun the main-results table with multiple seeds and report variance.

The script reuses the method registry and training functions from
``run_single_method.py``. Hyperparameters are read from existing
``results/<Env>/<question>/params_*.json`` files when available, with runner
defaults used for older/missing params files.
"""

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments_runner import ExperimentResults  # noqa: E402
from run_single_method import METHODS, result_filename  # noqa: E402


ENVS = ["CartPole", "Acrobot", "LunarLander", "MountainCar"]
CONDITIONS = {
    "homo": ("q1", "q1_homogeneous"),
    "hetero": ("q2", "q2_heterogeneous"),
}

TABLE_METHODS = [
    ("independent_qhd", "independent_qhd_hetero", "Indep. QHD"),
    ("truncate_fedavg_qhd", "truncate_fedavg_qhd_hetero", "Trunc. FedAvg-QHD"),
    ("fedavg_dqn", None, "FedAvg-DQN"),
    ("distillation_dqn", "distillation_dqn_hetero", "Distill. FedDQN"),
    ("fedqhd_homo", "fedqhd_hetero", "FedQHD"),
    ("oracle_qhd", "oracle_qhd_hetero", "Oracle QHD$^\\dagger$"),
    ("oracle_dqn", "oracle_dqn_hetero", "Oracle DQN$^\\dagger$"),
]

DEFAULTS = {
    "episodes": 600,
    "agent_num": 5,
    "runs": 1,
    "random_seed": 42,
    "qhd_lr": 0.1,
    "qhd_discount": 0.95,
    "qhd_exploration_rate": 1.0,
    "qhd_exploration_decay": 0.95,
    "qhd_exploration_min": 0.001,
    "qhd_agg_interval": 25,
    "dqn_lr": 0.001,
    "dqn_discount": 0.99,
    "dqn_exploration_rate": 1.0,
    "dqn_exploration_decay": 0.995,
    "dqn_exploration_min": 0.001,
    "dqn_agg_interval": 25,
    "hyperdimension": 10000,
    "rff_gamma": 1.0,
    "anchor_set_size": 200,
    "hetero_dims": None,
    "output_dir": "results",
    "grid_size": [8, 8],
}


def params_path(source_results: Path, env: str, condition: str, method_key: str) -> Path:
    _, subdir = CONDITIONS[condition]
    return source_results / env / subdir / f"params_{method_key}.json"


def load_params(source_results: Path, env: str, condition: str, method_key: str) -> dict:
    path = params_path(source_results, env, condition, method_key)
    if not path.exists() and method_key == "truncate_fedavg_qhd_hetero":
        legacy_path = params_path(source_results, env, condition, "truncate_fedavg_qhd")
        if legacy_path.exists():
            path = legacy_path
    if path.exists():
        with path.open() as f:
            params = json.load(f)
    else:
        params = {}

    normalized = dict(DEFAULTS)
    normalized.update(params)
    normalized["method"] = method_key
    normalized["env"] = env
    normalized["question"] = CONDITIONS[condition][0]
    normalized["method_key"] = method_key

    # Older params files used these generic names.
    if "aggregation_interval" in params:
        if method_key in {"fedavg_dqn", "oracle_dqn", "oracle_dqn_hetero",
                          "distillation_dqn", "distillation_dqn_hetero"}:
            normalized["dqn_agg_interval"] = params["aggregation_interval"]
        else:
            normalized["qhd_agg_interval"] = params["aggregation_interval"]

    if normalized.get("dqn_lr") is None:
        normalized["dqn_lr"] = DEFAULTS["dqn_lr"]
    if normalized.get("dqn_discount") is None:
        normalized["dqn_discount"] = DEFAULTS["dqn_discount"]
    if normalized.get("dqn_exploration_rate") is None:
        normalized["dqn_exploration_rate"] = DEFAULTS["dqn_exploration_rate"]
    if normalized.get("dqn_exploration_decay") is None:
        normalized["dqn_exploration_decay"] = DEFAULTS["dqn_exploration_decay"]
    if normalized.get("dqn_exploration_min") is None:
        normalized["dqn_exploration_min"] = DEFAULTS["dqn_exploration_min"]

    normalized["source_params"] = str(path) if path.exists() else None
    return normalized


def average_results(results: list[ExperimentResults], display_name: str) -> ExperimentResults:
    averaged = ExperimentResults()
    averaged.method_name = display_name

    max_len = max(len(r.reward_history) for r in results)
    pad = lambda xs: xs + [xs[-1]] * (max_len - len(xs))
    averaged.reward_history = np.mean([pad(r.reward_history) for r in results], axis=0).tolist()
    averaged.success_history = np.mean([pad(r.success_history) for r in results], axis=0).tolist()
    averaged.training_time = float(np.mean([r.training_time for r in results]))
    averaged.final_avg_reward = float(np.mean([r.final_avg_reward for r in results]))
    averaged.final_avg_success = float(np.mean([r.final_avg_success for r in results]))
    convergence = [r.convergence_episode for r in results if r.convergence_episode is not None]
    averaged.convergence_episode = float(np.mean(convergence)) if convergence else None

    residuals = [r.projection_residuals for r in results if r.projection_residuals]
    if residuals:
        max_rlen = max(len(r) for r in residuals)
        pad_r = lambda xs: xs + [xs[-1]] * (max_rlen - len(xs))
        averaged.projection_residuals = np.mean([pad_r(r) for r in residuals], axis=0).tolist()

    return averaged


def run_config(params: dict, seeds: list[int], output_root: Path) -> dict:
    display_name, train_fn = METHODS[params["method"]]
    per_seed = []

    for seed in seeds:
        args = SimpleNamespace(**{k: v for k, v in params.items() if k != "source_params"})
        args.random_seed = seed
        args.output_dir = str(output_root)
        args.runs = 1
        np.random.seed(seed)

        print(f"[{time.strftime('%H:%M:%S')}] {args.env} {args.question} {args.method} seed={seed}")
        result = train_fn(args.episodes, args)
        per_seed.append({"seed": seed, "result": result})

    rewards = np.array([item["result"].final_avg_reward for item in per_seed], dtype=float)
    successes = np.array([item["result"].final_avg_success for item in per_seed], dtype=float)
    times = np.array([item["result"].training_time for item in per_seed], dtype=float)
    averaged = average_results([item["result"] for item in per_seed], display_name)

    subdir = "q1_homogeneous" if params["question"] == "q1" else "q2_heterogeneous"
    out_dir = output_root / params["env"] / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    result_data = averaged.to_dict()
    result_data["seed_rewards"] = rewards.tolist()
    result_data["seed_successes"] = successes.tolist()
    result_data["seed_training_times"] = times.tolist()
    result_data["reward_std"] = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    result_data["reward_variance"] = float(np.var(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    result_data["success_std"] = float(np.std(successes, ddof=1)) if len(successes) > 1 else 0.0
    result_data["success_variance"] = float(np.var(successes, ddof=1)) if len(successes) > 1 else 0.0
    result_data["seeds"] = seeds
    result_data["source_params"] = params.get("source_params")

    with (out_dir / result_filename(display_name)).open("w") as f:
        json.dump(result_data, f, indent=2)

    params_out = dict(params)
    params_out["runs"] = len(seeds)
    params_out["seeds"] = seeds
    with (out_dir / f"params_{params['method']}.json").open("w") as f:
        json.dump(params_out, f, indent=2)

    return {
        "env": params["env"],
        "condition": "homo" if params["question"] == "q1" else "hetero",
        "method_key": params["method"],
        "method_name": display_name,
        "source_params": params.get("source_params"),
        "seeds": seeds,
        "seed_rewards": rewards.tolist(),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "variance": float(np.var(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "training_time_mean": float(np.mean(times)),
    }


def load_existing_record(params: dict, seeds: list[int], output_root: Path) -> dict | None:
    display_name, _ = METHODS[params["method"]]
    subdir = "q1_homogeneous" if params["question"] == "q1" else "q2_heterogeneous"
    path = output_root / params["env"] / subdir / result_filename(display_name)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    if data.get("seeds") != seeds or "seed_rewards" not in data:
        return None
    rewards = np.array(data["seed_rewards"], dtype=float)
    return {
        "env": params["env"],
        "condition": "homo" if params["question"] == "q1" else "hetero",
        "method_key": params["method"],
        "method_name": display_name,
        "source_params": data.get("source_params", params.get("source_params")),
        "seeds": seeds,
        "seed_rewards": rewards.tolist(),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "variance": float(np.var(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "training_time_mean": float(np.mean(data.get("seed_training_times", [0.0]))),
    }


def fmt_cell(record: dict | None, latex: bool = False) -> str:
    if record is None:
        return "{--}" if latex else "--"
    value = f"{record['mean']:.2f} $\\pm$ {record['std']:.2f}" if latex else f"{record['mean']:.2f} ± {record['std']:.2f}"
    return value


def write_tables(records: list[dict], output_root: Path) -> None:
    by_key = {(r["method_key"], r["env"], r["condition"]): r for r in records}

    md = [
        "# Main Results With 3-Seed Variance",
        "",
        "Final average reward over the last 100 episodes, reported as mean ± sample std across seeds. The JSON summary also includes sample variance.",
        "",
        "| Method | CartPole Homo | CartPole Hetero | Acrobot Homo | Acrobot Hetero | LunarLander Homo | LunarLander Hetero | MountainCar Homo | MountainCar Hetero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for homo_key, hetero_key, label in TABLE_METHODS:
        row = [label.replace("$^\\dagger$", "†")]
        for env in ENVS:
            row.append(fmt_cell(by_key.get((homo_key, env, "homo"))))
            row.append(fmt_cell(by_key.get((hetero_key, env, "hetero")) if hetero_key else None))
        md.append("| " + " | ".join(row) + " |")
    (output_root / "main_results_3seed_variance.md").write_text("\n".join(md) + "\n")

    tex = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\small",
        "\\caption{Final average reward (last 100 episodes, $N{=}5$), mean $\\pm$ sample standard deviation across 3 seeds.}",
        "\\label{tab:main_results_3seed_variance}",
        "\\begin{tabular}{l rr rr rr rr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{CartPole} & \\multicolumn{2}{c}{Acrobot} & \\multicolumn{2}{c}{LunarLander} & \\multicolumn{2}{c}{MountainCar} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Method & Homo & Hetero & Homo & Hetero & Homo & Hetero & Homo & Hetero \\\\",
        "\\midrule",
    ]
    for homo_key, hetero_key, label in TABLE_METHODS:
        cells = []
        for env in ENVS:
            cells.append(fmt_cell(by_key.get((homo_key, env, "homo")), latex=True))
            cells.append(fmt_cell(by_key.get((hetero_key, env, "hetero")) if hetero_key else None, latex=True))
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if homo_key == "fedqhd_homo":
            tex.append("\\midrule")
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_root / "main_results_3seed_variance.tex").write_text("\n".join(tex))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", default="results", type=Path)
    parser.add_argument("--output-root", default=Path("results/main_table_3seed"), type=Path)
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds.")
    parser.add_argument("--env", choices=ENVS, action="append", help="Limit to one or more environments.")
    parser.add_argument("--method", choices=sorted(METHODS.keys()), action="append", help="Limit to one or more method keys.")
    parser.add_argument("--condition", choices=["homo", "hetero"], action="append", help="Limit to homo/q1 or hetero/q2.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed result JSON files with matching seeds.")
    parser.add_argument("--jobs", default=1, type=int, help="Number of configs to run in parallel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    envs = args.env or ENVS
    conditions = args.condition or ["homo", "hetero"]
    selected_methods = set(args.method) if args.method else None
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    work_items = []
    for env in envs:
        for condition in conditions:
            for homo_key, hetero_key, _ in TABLE_METHODS:
                method_key = homo_key if condition == "homo" else hetero_key
                if method_key is None:
                    continue
                if selected_methods and method_key not in selected_methods:
                    continue
                params = load_params(args.source_results, env, condition, method_key)
                work_items.append(params)

    records = []
    pending = []
    if args.resume:
        for params in work_items:
            record = load_existing_record(params, seeds, output_root)
            if record is not None:
                print(f"[{time.strftime('%H:%M:%S')}] reuse {params['env']} {params['question']} {params['method']}")
                records.append(record)
            else:
                pending.append(params)
    else:
        pending = work_items

    def persist_partial() -> None:
        write_tables(records, output_root)
        with (output_root / "variance_summary.json").open("w") as f:
            json.dump({
                "seeds": seeds,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "records": records,
            }, f, indent=2)

    persist_partial()

    if args.jobs == 1:
        for params in pending:
            records.append(run_config(params, seeds, output_root))
            persist_partial()
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_config, params, seeds, output_root) for params in pending]
            for future in as_completed(futures):
                records.append(future.result())
                persist_partial()

    write_tables(records, output_root)
    with (output_root / "variance_summary.json").open("w") as f:
        json.dump({
            "seeds": seeds,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": records,
        }, f, indent=2)

    print(f"\nWrote summary: {output_root / 'variance_summary.json'}")
    print(f"Wrote table:   {output_root / 'main_results_3seed_variance.md'}")
    print(f"Wrote LaTeX:   {output_root / 'main_results_3seed_variance.tex'}")


if __name__ == "__main__":
    main()
