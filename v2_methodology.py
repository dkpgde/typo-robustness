"""Methodologically strict utilities for Typo Robustness v2.

This module is the authoritative layer for the adaptive-tokenization study.
It deliberately sits on top of ``common.py`` so the frozen v1 training and
corruption machinery remains unchanged while v2 evaluation is leakage-safe.

Key invariants
--------------
* Expert training remains exactly the frozen v1 protocol.
* Router fitting never uses the final test examples.
* OOD corruption holdout is a real holdout: the held-out family is excluded
  from router calibration and used only for OOD evaluation.
* Learned routers are fit separately for every lambda in the predefined sweep.
* Frontier cost is measured end-to-end CPU latency, including tokenization,
  instability-feature computation and router inference where applicable.
* Routing regret is defined against the same lambda-penalized oracle objective
  that generated the routing target.
"""

from __future__ import annotations

import html
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from common import (
    CORRUPTIONS,
    CORRUPTION_LEVELS,
    DATASET_PATH,
    EXPERT_SEQUENCE_LENGTHS,
    LAMBDA_SWEEP,
    NUM_CLASSES,
    ROUTER_EXPERTS,
    ROUTER_FEATURES,
    TRAIN_CSV,
    TRAINING_SEEDS,
    build_expert_probability_table,
    build_nested_test_sets,
    compute_instability_features,
    fit_learned_router,
    fragmentation_threshold_router,
    route_probabilities,
)

ROUTER_CALIBRATION_PER_CLASS = 500
OOD_HOLDOUT_FAMILY = "substitution"
# coarse -> fine. Smaller BPE vocabularies produce finer segmentation.
GRANULARITY_ORDER = {
    "word": 0,
    "bpe_5000": 1,
    "bpe_2000": 2,
    "bpe_1000": 3,
    "bpe_500": 4,
    "char": 5,
}


def _clean_description(series: pd.Series) -> pd.Series:
    """Apply the same text cleanup semantics used by the frozen v1 pipeline."""
    return (
        series.map(html.unescape)
        .str.replace("\\", " ", regex=False)
        .str.replace("quot;", "", regex=False)
        .str.replace("--", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_router_calibration_set(per_class: int = ROUTER_CALIBRATION_PER_CLASS) -> pd.DataFrame:
    """Load examples disjoint from v1 train/validation and final test.

    v1 consumes the first 3,500 examples per class from AG News ``train.csv``.
    Router calibration uses the *next* ``per_class`` examples per class. This
    preserves the frozen expert experiment while avoiding the previous error
    of fitting the router on the final test inputs under different model seeds.
    """
    raw = pd.read_csv(TRAIN_CSV)
    grouped = raw.groupby("Class Index", sort=False)
    frames = []
    for _, group in grouped:
        frames.append(group.iloc[3500 : 3500 + per_class].copy())
    calibration = pd.concat(frames, axis=0).drop(columns=["Title"])
    if len(calibration) != per_class * calibration["Class Index"].nunique():
        raise ValueError("AG News does not contain enough unused rows for router calibration.")
    calibration["Description"] = _clean_description(calibration["Description"])
    # Keep stable IDs distinct from the final test IDs and from v1 train IDs.
    calibration = calibration.reset_index(drop=False).rename(columns={"index": "source_index"})
    calibration.index = pd.Index(np.arange(len(calibration)), name="sample_id")
    return calibration


def build_router_calibration_sets(
    calibration_df: pd.DataFrame,
    holdout_family: str | None = None,
    corruption_seed: int = 9173,
):
    """Build nested calibration corruptions, optionally excluding one family."""
    allowed = None
    if holdout_family is not None:
        if holdout_family not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption family: {holdout_family}")
        allowed = [c for c in CORRUPTIONS if c != holdout_family]
    return build_nested_test_sets(
        source_df=calibration_df,
        corruption_levels=CORRUPTION_LEVELS,
        corruption_seed=corruption_seed,
        allowed_corruptions=allowed,
    )


def build_true_ood_sets(test_df: pd.DataFrame, holdout_family: str = OOD_HOLDOUT_FAMILY):
    """Build held-out-family-only test sets for the OOD evaluation."""
    if holdout_family not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption family: {holdout_family}")
    return build_nested_test_sets(
        source_df=test_df,
        corruption_levels=CORRUPTION_LEVELS,
        corruption_seed=0,
        allowed_corruptions=[holdout_family],
    )


def instability_frame(test_sets, suite, word_vocab) -> pd.DataFrame:
    frames = []
    for level, frame in test_sets.items():
        features = compute_instability_features(frame["Description"].to_numpy(), suite, word_vocab)
        features.insert(0, "corruption_level", level)
        features.insert(0, "sample_id", frame.index.to_numpy())
        frames.append(features)
    out = pd.concat(frames, ignore_index=True)
    return add_clean_counterpart_deltas(out)


def add_clean_counterpart_deltas(features: pd.DataFrame) -> pd.DataFrame:
    """Add paired clean/corrupted deltas for analysis only, never router inputs."""
    clean = features[features["corruption_level"].eq(0)][
        ["sample_id", "bpe_tokens_per_word", "sequence_length"]
    ].rename(
        columns={
            "bpe_tokens_per_word": "clean_bpe_tokens_per_word",
            "sequence_length": "clean_sequence_length",
        }
    )
    out = features.merge(clean, on="sample_id", how="left", validate="many_to_one")
    out["delta_fragmentation"] = out["bpe_tokens_per_word"] - out["clean_bpe_tokens_per_word"]
    out["delta_sequence_length"] = out["sequence_length"] - out["clean_sequence_length"]
    return out


def router_feature_rows(prob_table: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Align deployable router features with expert predictions without seed leakage."""
    merged = prob_table[["sample_id", "seed", "corruption_level"]].merge(
        features,
        on=["sample_id", "corruption_level"],
        how="left",
        validate="many_to_one",
    )
    if merged[ROUTER_FEATURES].isna().any().any():
        raise ValueError("Missing instability features after alignment.")
    return merged


def expert_loss_matrix(prob_table: pd.DataFrame, experts: Sequence[str] = ROUTER_EXPERTS) -> np.ndarray:
    y = prob_table["true_class"].to_numpy(dtype=int) - 1
    take = np.arange(len(prob_table))
    losses = []
    for expert in experts:
        probs = prob_table[[f"prob_class_{c}__{expert}" for c in range(1, NUM_CLASSES + 1)]].to_numpy()
        losses.append(-np.log(np.clip(probs[take, y], 1e-12, 1.0)))
    return np.column_stack(losses)


def oracle_targets(
    prob_table: pd.DataFrame,
    expert_cost_ms: dict[str, float],
    lambdas: Sequence[float] = LAMBDA_SWEEP,
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> pd.DataFrame:
    """Return the exact lambda-penalized oracle target for every row."""
    losses = expert_loss_matrix(prob_table, experts)
    costs = np.asarray([expert_cost_ms[e] for e in experts], dtype=float)
    key = prob_table[["sample_id", "seed", "corruption_level"]].reset_index(drop=True)
    frames = []
    for lam in lambdas:
        objective = losses + float(lam) * costs[None, :]
        choice = objective.argmin(axis=1)
        part = key.copy()
        part["lambda"] = float(lam)
        part["oracle_expert"] = [experts[i] for i in choice]
        part["oracle_objective"] = objective[np.arange(len(objective)), choice]
        part["oracle_predictive_loss"] = losses[np.arange(len(losses)), choice]
        part["oracle_cost_ms"] = costs[choice]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def fit_router_sweep(
    calibration_features: pd.DataFrame,
    calibration_oracle: pd.DataFrame,
    router_type: str = "logistic_regression",
    lambdas: Sequence[float] = LAMBDA_SWEEP,
) -> dict[float, object]:
    """Fit a separate seed-agnostic router for every lambda on calibration examples."""
    key_cols = ["sample_id", "seed", "corruption_level"]
    base = calibration_features[key_cols + ROUTER_FEATURES].reset_index(drop=True)
    models = {}
    for lam in lambdas:
        labels = calibration_oracle[calibration_oracle["lambda"].eq(float(lam))][
            key_cols + ["oracle_expert"]
        ]
        aligned = base.merge(labels, on=key_cols, how="inner", validate="one_to_one")
        models[float(lam)] = fit_learned_router(
            aligned[ROUTER_FEATURES],
            aligned["oracle_expert"].to_numpy(),
            router_type=router_type,
            random_state=0,
        )
    return models


def predict_router(estimator, feature_rows: pd.DataFrame) -> np.ndarray:
    """Predict with either sklearn pipeline-style or raw estimators safely."""
    try:
        return np.asarray(estimator.predict(feature_rows[ROUTER_FEATURES]))
    except Exception:
        return np.asarray(estimator.predict(feature_rows[ROUTER_FEATURES].to_numpy()))


def lambda_routing_regret(
    routed: pd.DataFrame,
    oracle: pd.DataFrame,
    expert_cost_ms: dict[str, float],
    lam: float,
) -> pd.DataFrame:
    """Regret against the *same lambda objective* used to define the oracle."""
    key = ["sample_id", "seed", "corruption_level"]
    target = oracle[oracle["lambda"].eq(float(lam))][
        key + ["oracle_expert", "oracle_objective", "oracle_predictive_loss", "oracle_cost_ms"]
    ]
    out = routed.merge(target, on=key, how="left", validate="one_to_one")
    chosen_cost = out["chosen_expert"].map(expert_cost_ms).astype(float)
    out["chosen_cost_ms"] = chosen_cost
    out["chosen_objective"] = out["log_loss_contrib"] + float(lam) * chosen_cost
    out["regret"] = out["chosen_objective"] - out["oracle_objective"]
    out["is_oracle_choice"] = out["chosen_expert"].eq(out["oracle_expert"])
    return out


def confidence_failure_diagnostics(
    prob_table: pd.DataFrame,
    features: pd.DataFrame,
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> pd.DataFrame:
    """Per-seed diagnostics for error, confidence loss and expert loss gaps."""
    rows = prob_table[["sample_id", "seed", "corruption_level", "true_class"]].copy()
    y = rows["true_class"].to_numpy(dtype=int) - 1
    take = np.arange(len(rows))
    losses = {}
    for expert in experts:
        pcols = [f"prob_class_{c}__{expert}" for c in range(1, NUM_CLASSES + 1)]
        probs = prob_table[pcols].to_numpy()
        pred = probs.argmax(axis=1)
        rows[f"correct_{expert}"] = (pred == y).astype(int)
        rows[f"confidence_{expert}"] = probs.max(axis=1)
        losses[expert] = -np.log(np.clip(probs[take, y], 1e-12, 1.0))
        rows[f"loss_{expert}"] = losses[expert]

    clean_cols = ["sample_id", "seed"] + [f"confidence_{e}" for e in experts]
    clean = rows[rows["corruption_level"].eq(0)][clean_cols].rename(
        columns={f"confidence_{e}": f"clean_confidence_{e}" for e in experts}
    )
    rows = rows.merge(clean, on=["sample_id", "seed"], how="left", validate="many_to_one")
    for expert in experts:
        rows[f"confidence_degradation_{expert}"] = (
            rows[f"clean_confidence_{expert}"] - rows[f"confidence_{expert}"]
        )
    if "word" in experts and "char" in experts:
        rows["loss_gap_word_minus_char"] = rows["loss_word"] - rows["loss_char"]
    return rows.merge(features, on=["sample_id", "corruption_level"], how="left", validate="many_to_one")


def diagnostic_correlations(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Correlate deployable instability signals with the three planned targets."""
    from scipy.stats import pointbiserialr, spearmanr

    rows = []
    for feature in ROUTER_FEATURES:
        valid = diagnostics[[feature, "correct_word"]].dropna()
        r, p = pointbiserialr(valid["correct_word"], valid[feature])
        rows.append({"feature": feature, "target": "word_error", "correlation": -float(r), "p_value": float(p)})
        for target in ("confidence_degradation_word", "loss_gap_word_minus_char"):
            if target not in diagnostics:
                continue
            valid = diagnostics[[feature, target]].dropna()
            stat = spearmanr(valid[feature], valid[target])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": float(stat.statistic),
                    "p_value": float(stat.pvalue),
                }
            )
    return pd.DataFrame(rows)


def _median_ms(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def measure_end_to_end_expert_latency_ms(
    model,
    suite,
    model_name: str,
    sample_texts: Sequence[str],
    n_repeat: int = 3,
) -> float:
    """Measure tokenization/vectorization + model inference, batch size one."""
    texts = list(sample_texts)[:64]
    timings = []
    for text in texts:
        # Warmup includes both path components.
        for _ in range(2):
            x = suite.vectorize(model_name, [text])
            _ = model.predict(x, batch_size=1, verbose=0)
        samples = []
        for _ in range(n_repeat):
            start = time.perf_counter()
            x = suite.vectorize(model_name, [text])
            _ = model.predict(x, batch_size=1, verbose=0)
            samples.append((time.perf_counter() - start) * 1000)
        timings.append(_median_ms(samples))
    return _median_ms(timings)


def measure_router_overhead_ms(
    estimator,
    suite,
    word_vocab,
    sample_texts: Sequence[str],
    n_repeat: int = 3,
) -> float:
    """Measure deployable instability-feature extraction + router prediction."""
    texts = list(sample_texts)[:64]
    timings = []
    for text in texts:
        samples = []
        for _ in range(n_repeat):
            start = time.perf_counter()
            feats = compute_instability_features([text], suite, word_vocab)
            _ = predict_router(estimator, feats)
            samples.append((time.perf_counter() - start) * 1000)
        timings.append(_median_ms(samples))
    return _median_ms(timings)


def measure_threshold_overhead_ms(
    suite,
    word_vocab,
    sample_texts: Sequence[str],
    thresholds: Sequence[float],
    n_repeat: int = 3,
) -> float:
    texts = list(sample_texts)[:64]
    timings = []
    for text in texts:
        samples = []
        for _ in range(n_repeat):
            start = time.perf_counter()
            feats = compute_instability_features([text], suite, word_vocab)
            _ = fragmentation_threshold_router(feats, thresholds=thresholds, experts=ROUTER_EXPERTS)
            samples.append((time.perf_counter() - start) * 1000)
        timings.append(_median_ms(samples))
    return _median_ms(timings)


def summarize_frontier_point(
    routed: pd.DataFrame,
    expert_latency_ms: dict[str, float],
    router_overhead_ms: float = 0.0,
    expert_parameters: dict[str, int] | None = None,
) -> dict:
    """Frontier summary using actual CPU cost rather than sequence length proxy."""
    from sklearn.metrics import f1_score

    expert_cost = routed["chosen_expert"].map(expert_latency_ms).astype(float)
    result = {
        "macro_f1": float(f1_score(routed["true_class"], routed["predicted_class"], average="macro", zero_division=0)),
        "mean_log_loss": float(routed["log_loss_contrib"].mean()),
        "mean_active_sequence_length": float(routed["active_sequence_length"].mean()),
        "mean_expert_latency_ms": float(expert_cost.mean()),
        "router_overhead_ms": float(router_overhead_ms),
        "mean_total_cpu_latency_ms": float(expert_cost.mean() + router_overhead_ms),
    }
    if expert_parameters:
        result["mean_active_parameters"] = float(routed["chosen_expert"].map(expert_parameters).mean())
    result.update({f"fraction_{e}": float(routed["chosen_expert"].eq(e).mean()) for e in ROUTER_EXPERTS})
    return result


def relative_f1_degradation(routed: pd.DataFrame) -> pd.DataFrame:
    """Compute seed-wise clean-relative Macro-F1 degradation by corruption level."""
    from sklearn.metrics import f1_score

    rows = []
    for seed, seed_df in routed.groupby("seed"):
        f1_by_level = {}
        for level, frame in seed_df.groupby("corruption_level"):
            f1_by_level[int(level)] = float(
                f1_score(frame["true_class"], frame["predicted_class"], average="macro", zero_division=0)
            )
        clean = f1_by_level[0]
        for level, f1 in f1_by_level.items():
            rows.append(
                {
                    "seed": seed,
                    "corruption_level": level,
                    "macro_f1": f1,
                    "relative_macro_f1_degradation": 0.0 if level == 0 else (clean - f1) / clean,
                }
            )
    return pd.DataFrame(rows)


def tune_threshold_router(
    calibration_prob: pd.DataFrame,
    calibration_features: pd.DataFrame,
    expert_cost_ms: dict[str, float],
    lam: float,
    grid: Iterable[float] = tuple(np.arange(0.02, 0.62, 0.04)),
) -> tuple[tuple[float, float], float]:
    """Tune a two-threshold word->BPE500->char router on calibration data only."""
    best = None
    best_score = np.inf
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            routing = fragmentation_threshold_router(
                calibration_features[ROUTER_FEATURES],
                thresholds=(float(low), float(high)),
                experts=ROUTER_EXPERTS,
            )
            routed = route_probabilities(calibration_prob, np.asarray(routing))
            costs = routed["chosen_expert"].map(expert_cost_ms).astype(float)
            score = float((routed["log_loss_contrib"] + float(lam) * costs).mean())
            if score < best_score:
                best_score = score
                best = (float(low), float(high))
    if best is None:
        raise RuntimeError("Threshold tuning grid did not produce a valid policy.")
    return best, best_score


def serialized_estimator_size_mb(estimator) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name
    try:
        with open(path, "wb") as handle:
            pickle.dump(estimator, handle)
        return os.path.getsize(path) / 1024**2
    finally:
        if os.path.exists(path):
            os.remove(path)


def paired_seed_effects(
    degradation: pd.DataFrame,
    adaptive_name: str,
    fixed_name: str,
) -> dict:
    """Paired seed-level effect size and 95% CI for mean relative degradation."""
    from scipy.stats import t

    sub = degradation[degradation["system"].isin([adaptive_name, fixed_name]) & degradation["corruption_level"].gt(0)]
    means = sub.groupby(["system", "seed"])["relative_macro_f1_degradation"].mean().unstack("system")
    diff = means[adaptive_name] - means[fixed_name]
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    crit = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
    return {
        "adaptive": adaptive_name,
        "fixed": fixed_name,
        "n_seeds": n,
        "mean_paired_difference": mean,
        "ci95_low": mean - crit * se if n > 1 else np.nan,
        "ci95_high": mean + crit * se if n > 1 else np.nan,
        "cohens_dz": mean / sd if n > 1 and sd > 0 else np.nan,
    }


def resident_system_size_mb(model_paths: dict[str, str], router=None) -> float:
    """Resident disk-size proxy: all stored experts plus the router artifact."""
    size = sum(os.path.getsize(model_paths[e]) for e in ROUTER_EXPERTS) / 1024**2
    if router is not None:
        size += serialized_estimator_size_mb(router)
    return float(size)
