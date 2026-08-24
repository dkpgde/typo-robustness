"""Correct CNN-vs-TCN analysis for authoritative v2 runs.

Consumes the committed artifacts produced by ``run_v2_authoritative.py``.
No training occurs here. The comparison is qualitative/matched-effect-size
replication, not an architecture horse race.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import FIGURES_DIR, RESULTS_DIR
from v2_methodology import GRANULARITY_ORDER


def latest_authoritative_run(backbone: str) -> str:
    manifests = sorted(glob.glob(os.path.join(RESULTS_DIR, f"run_manifest_{backbone}_*.json")))
    valid = []
    for path in manifests:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("protocol") == "v2_authoritative_leakage_safe":
            valid.append(data["run_id"])
    if not valid:
        raise FileNotFoundError(f"No authoritative v2 manifest found for {backbone}.")
    return valid[-1]


def load(backbone: str, run_id: str) -> dict[str, pd.DataFrame]:
    mapping = {
        "metrics": f"{backbone}_metrics_{run_id}.csv",
        "features": f"{backbone}_instability_features_{run_id}.csv",
        "frontier": f"{backbone}_routing_frontier_{run_id}.csv",
        "ood_frontier": f"{backbone}_ood_routing_frontier_{run_id}.csv",
        "diagnostics": f"{backbone}_{run_id}_failure_diagnostics.csv",
        "correlations": f"{backbone}_{run_id}_diagnostic_correlations.csv",
        "effects": f"{backbone}_paired_effects_{run_id}.csv",
        "failure_modes": f"{backbone}_failure_mode_counts_{run_id}.csv",
        "relative_degradation": f"{backbone}_relative_degradation_{run_id}.csv",
    }
    out = {}
    for key, name in mapping.items():
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            out[key] = pd.read_csv(path)
    return out


def clean_vs_degradation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Seed-correct clean performance and degradation; never uses .first()."""
    per_seed_clean = (
        metrics[metrics["corruption_level"].eq(0)]
        .groupby(["model", "seed"], as_index=False)["f1_macro"]
        .mean()
        .rename(columns={"f1_macro": "clean_f1"})
    )
    corrupted = (
        metrics[metrics["corruption_level"].gt(0)]
        .groupby(["model", "seed"], as_index=False)["f1_macro"]
        .mean()
        .rename(columns={"f1_macro": "mean_corrupted_f1"})
    )
    merged = per_seed_clean.merge(corrupted, on=["model", "seed"], validate="one_to_one")
    merged["relative_drop"] = (merged["clean_f1"] - merged["mean_corrupted_f1"]) / merged["clean_f1"]
    return merged.groupby("model", as_index=True).agg(
        clean_f1=("clean_f1", "mean"),
        mean_corrupted_f1=("mean_corrupted_f1", "mean"),
        relative_drop=("relative_drop", "mean"),
        relative_drop_sd=("relative_drop", "std"),
    )


def granularity_correlations(metrics: pd.DataFrame) -> dict[str, float]:
    table = clean_vs_degradation(metrics)
    ranks = np.asarray([GRANULARITY_ORDER[m] for m in table.index], dtype=float)
    # GRANULARITY_ORDER is explicitly coarse -> fine:
    # word, BPE-5000, BPE-2000, BPE-1000, BPE-500, char.
    return {
        "spearman_fineness_vs_relative_drop": float(spearmanr(ranks, table["relative_drop"]).statistic),
        "spearman_fineness_vs_clean_f1": float(spearmanr(ranks, table["clean_f1"]).statistic),
    }


def best_adaptive_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    learned = frontier[frontier["system"].eq("learned_logistic_regression")].copy()
    fixed = frontier[frontier["system"].str.startswith("fixed_")].copy()
    rows = []
    for _, a in learned.iterrows():
        affordable = fixed[fixed["mean_total_cpu_latency_ms"].le(a["mean_total_cpu_latency_ms"])]
        if affordable.empty:
            best_f1 = np.nan
            best_name = None
        else:
            best = affordable.loc[affordable["macro_f1"].idxmax()]
            best_f1 = float(best["macro_f1"])
            best_name = best["system"]
        rows.append(
            {
                "lambda": a["lambda"],
                "adaptive_f1": a["macro_f1"],
                "adaptive_cpu_ms": a["mean_total_cpu_latency_ms"],
                "best_fixed_at_or_below_cost": best_name,
                "best_fixed_f1_at_or_below_cost": best_f1,
                "f1_gain_at_matched_cost": a["macro_f1"] - best_f1 if np.isfinite(best_f1) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def router_allocation(frontier: pd.DataFrame) -> pd.DataFrame:
    cols = ["lambda", "fraction_word", "fraction_bpe_500", "fraction_char"]
    return frontier[frontier["system"].eq("learned_logistic_regression")][cols].sort_values("lambda")


def save_two_panel_frontier(cnn: pd.DataFrame, tcn: pd.DataFrame, output: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, backbone, frontier in zip(axes, ["CNN", "TCN"], [cnn, tcn]):
        for system, sub in frontier.groupby("system"):
            if system.startswith("learned_") or system in {"oracle", "fragmentation_threshold"}:
                sub = sub.sort_values("mean_total_cpu_latency_ms")
                ax.plot(sub["mean_total_cpu_latency_ms"], sub["macro_f1"], marker="o", label=system)
            else:
                ax.scatter(sub["mean_total_cpu_latency_ms"], sub["macro_f1"], label=system)
        ax.set_title(backbone)
        ax.set_xlabel("Mean end-to-end CPU latency (ms/input)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cnn_run = latest_authoritative_run("cnn")
    tcn_run = latest_authoritative_run("tcn")
    cnn = load("cnn", cnn_run)
    tcn = load("tcn", tcn_run)

    summary = {}
    for backbone, data in (("cnn", cnn), ("tcn", tcn)):
        summary[backbone] = {
            "run_id": cnn_run if backbone == "cnn" else tcn_run,
            "granularity": granularity_correlations(data["metrics"]),
        }
        clean_vs_degradation(data["metrics"]).to_csv(
            os.path.join(RESULTS_DIR, f"comparison_{backbone}_clean_vs_degradation.csv")
        )
        best_adaptive_frontier(data["frontier"]).to_csv(
            os.path.join(RESULTS_DIR, f"comparison_{backbone}_matched_cost_frontier.csv"), index=False
        )
        router_allocation(data["frontier"]).to_csv(
            os.path.join(RESULTS_DIR, f"comparison_{backbone}_router_allocation.csv"), index=False
        )

    # Matched effect-size table: same named quantities, not raw architecture ranking.
    effect_frames = []
    for backbone, data in (("cnn", cnn), ("tcn", tcn)):
        if "effects" in data:
            frame = data["effects"].copy()
            frame.insert(0, "backbone", backbone)
            effect_frames.append(frame)
    if effect_frames:
        pd.concat(effect_frames, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "comparison_matched_effect_sizes.csv"), index=False
        )

    # Instability signal replication using the correctly seed-aware diagnostics.
    corr_frames = []
    for backbone, data in (("cnn", cnn), ("tcn", tcn)):
        if "correlations" in data:
            frame = data["correlations"].copy()
            frame.insert(0, "backbone", backbone)
            corr_frames.append(frame)
    if corr_frames:
        pd.concat(corr_frames, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "comparison_instability_correlations.csv"), index=False
        )

    # Router allocation similarity is compared at matching lambdas.
    cnn_alloc = router_allocation(cnn["frontier"]).add_suffix("_cnn").rename(columns={"lambda_cnn": "lambda"})
    tcn_alloc = router_allocation(tcn["frontier"]).add_suffix("_tcn").rename(columns={"lambda_tcn": "lambda"})
    allocation = cnn_alloc.merge(tcn_alloc, on="lambda", validate="one_to_one")
    allocation.to_csv(os.path.join(RESULTS_DIR, "comparison_router_allocation_similarity.csv"), index=False)

    # Same failure taxonomy under both backbones.
    failures = []
    for backbone, data in (("cnn", cnn), ("tcn", tcn)):
        if "failure_modes" in data:
            frame = data["failure_modes"].copy()
            frame.insert(0, "backbone", backbone)
            failures.append(frame)
    if failures:
        pd.concat(failures, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "comparison_failure_modes.csv"), index=False
        )

    save_two_panel_frontier(
        cnn["frontier"],
        tcn["frontier"],
        os.path.join(FIGURES_DIR, "v2_cnn_vs_tcn_cpu_frontier.png"),
    )
    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
