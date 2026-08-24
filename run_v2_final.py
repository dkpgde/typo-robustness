"""Final execution pipeline for Typo Robustness v2.

Run once per backbone:
    python run_v2_final.py --backbone cnn
    python run_v2_final.py --backbone tcn

This file is authoritative for final v2 results. The v2 notebooks remain useful
for exploration, but must not be used to produce reported numbers.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os

import keras
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import common
from common import (
    ARTIFACTS_DIR,
    BPE_DROPOUT_PROB,
    BPE_SEQUENCE_LENGTHS,
    CORRUPTION_LEVELS,
    LAMBDA_SWEEP,
    RESULTS_DIR,
    ROUTER_EXPERTS,
    ROUTER_FEATURES,
    TRAINING_SEEDS,
    BPEDropoutTrainingSequence,
    PeakRAMMonitor,
    build_cnn,
    build_expert_probability_table,
    build_tcn,
    build_tokenization_suite,
    enable_determinism,
    ensure_dirs,
    evaluate_model,
    label_failure_modes,
    load_test_set,
    load_train_val_set,
    make_bpe_dropout_tokenizer,
    make_train_val_split,
    make_word_vocab_index,
    mcnemar_holm_table,
    new_run_id,
    route_probabilities,
    run_fixed_comparison,
    save_artifacts,
    save_json,
    save_split_ids,
    train_model,
)
from v2_methodology import (
    OOD_HOLDOUT_FAMILY,
    build_router_calibration_sets,
    build_true_ood_sets,
    confidence_failure_diagnostics,
    diagnostic_correlations,
    fit_router_sweep,
    instability_frame,
    lambda_routing_regret,
    load_router_calibration_set,
    measure_end_to_end_expert_latency_ms,
    measure_router_overhead_ms,
    measure_threshold_overhead_ms,
    oracle_targets,
    predict_router,
    relative_f1_degradation,
    resident_system_size_mb,
    router_feature_rows,
    summarize_frontier_point,
    tune_threshold_router,
)

FIXED_FRONTIER_MODELS = ["word", "bpe_500", "bpe_5000", "char"]
ROUTER_TYPES = ["logistic_regression", "decision_tree", "gradient_boosting"]
PRIMARY_ROUTER = "logistic_regression"


def model_path(backbone: str, model_name: str, seed: int, run_id: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "models", f"{backbone}_{model_name}_seed{seed}_{run_id}.keras")


def model_builder(backbone: str, sequence_length: int, vocab_size: int):
    if backbone == "cnn":
        return lambda: build_cnn(sequence_length, vocab_size, 32)
    return lambda: build_tcn(sequence_length, vocab_size)


def evaluate_saved(backbone, suite, datasets, run_id, model_names=ROUTER_EXPERTS):
    metrics, predictions = [], []
    for model_name in model_names:
        for seed in TRAINING_SEEDS:
            model = keras.models.load_model(model_path(backbone, model_name, seed, run_id))
            evaluate_model(
                model=model,
                model_name=model_name,
                seed=seed,
                suite=suite,
                test_sets=datasets,
                metric_rows=metrics,
                prediction_frames=predictions,
            )
            del model
            keras.backend.clear_session()
            gc.collect()
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def fixed_routed(predictions_long: pd.DataFrame, model_name: str) -> pd.DataFrame:
    table = build_expert_probability_table(predictions_long, experts=[model_name])
    return route_probabilities(
        table,
        np.asarray([model_name] * len(table)),
        experts=[model_name],
    )


def predictions_to_single_system_routed(predictions: pd.DataFrame, system: str, sequence_length: int) -> pd.DataFrame:
    sub = predictions[predictions["model"].eq(system)].copy()
    probability_columns = [f"prob_class_{i}" for i in range(1, common.NUM_CLASSES + 1)]
    probs = sub[probability_columns].to_numpy()
    predicted = probs.argmax(axis=1) + 1
    y = sub["true_class"].to_numpy(dtype=int)
    take = np.arange(len(sub))
    out = sub[["sample_id", "seed", "corruption_level"]].copy()
    out["true_class"] = y
    out["chosen_expert"] = system
    out["predicted_class"] = predicted
    out["correct"] = (predicted == y).astype(int)
    out["log_loss_contrib"] = -np.log(np.clip(probs[take, y - 1], 1e-12, 1.0))
    out["active_sequence_length"] = int(sequence_length)
    return out


def expert_parameter_map(resources: pd.DataFrame, names) -> dict[str, int]:
    return (
        resources[resources["model"].isin(names)]
        .groupby("model")["n_parameters"]
        .median()
        .astype(int)
        .to_dict()
    )


def measure_fixed_costs(backbone, suite, test_df, run_id, names):
    texts = test_df["Description"].iloc[:64].to_numpy()
    rows = []
    for name in names:
        vectorizer_name = "bpe_500" if name == "bpe_500_dropout" else name
        for seed in TRAINING_SEEDS:
            model = keras.models.load_model(model_path(backbone, name, seed, run_id))
            latency = measure_end_to_end_expert_latency_ms(model, suite, vectorizer_name, texts)
            rows.append({"model": name, "seed": seed, "end_to_end_latency_ms": latency})
            del model
            keras.backend.clear_session()
            gc.collect()
    frame = pd.DataFrame(rows)
    return frame, frame.groupby("model")["end_to_end_latency_ms"].mean().to_dict()


def train_bpe_dropout(backbone, suite, train_df, val_df, test_sets, run_id):
    tokenizer = make_bpe_dropout_tokenizer(suite.bpe_tokenizers[500], dropout_prob=BPE_DROPOUT_PROB)
    train_y = train_df["Class Index"].to_numpy() - 1
    val_y = val_df["Class Index"].to_numpy() - 1
    val_x = suite.vectorize("bpe_500", val_df["Description"].to_numpy())
    metrics, predictions, resources = [], [], []
    for seed in TRAINING_SEEDS:
        sequence = BPEDropoutTrainingSequence(train_df["Description"].to_numpy(), train_y, tokenizer)
        model, _, stats = train_model(
            model_builder=model_builder(backbone, BPE_SEQUENCE_LENGTHS[500], suite.bpe_tokenizers[500].get_vocab_size()),
            train_X=sequence,
            train_Y=None,
            val_X=val_x,
            val_Y=val_y,
            seed=seed,
        )
        evaluate_model(
            model=model,
            model_name="bpe_500",
            seed=seed,
            suite=suite,
            test_sets=test_sets,
            metric_rows=metrics,
            prediction_frames=predictions,
            system_label="bpe_500_dropout",
        )
        resources.append(
            {
                "model": "bpe_500_dropout",
                "seed": seed,
                "n_parameters": model.count_params(),
                "training_time_s": stats["training_time_s"],
                "peak_ram_mb": stats["peak_ram_mb"],
            }
        )
        model.save(model_path(backbone, "bpe_500_dropout", seed, run_id))
        del model
        keras.backend.clear_session()
        gc.collect()
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(resources)


def fit_routing_family(
    backbone,
    run_id,
    label,
    suite,
    word_vocab,
    calibration_prob,
    calibration_features,
    evaluation_prob,
    evaluation_features,
    expert_cost_ms,
    expert_parameters,
    latency_texts,
):
    calibration_rows = router_feature_rows(calibration_prob, calibration_features)
    evaluation_rows = router_feature_rows(evaluation_prob, evaluation_features)
    calibration_oracle = oracle_targets(calibration_prob, expert_cost_ms)
    evaluation_oracle = oracle_targets(evaluation_prob, expert_cost_ms)
    calibration_oracle.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_calibration_oracle_{run_id}.csv"), index=False)
    evaluation_oracle.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_evaluation_oracle_{run_id}.csv"), index=False)

    frontier_rows, routed_systems, regret_frames = [], {}, []

    for lam in LAMBDA_SWEEP:
        oracle_labels = evaluation_oracle[evaluation_oracle["lambda"].eq(float(lam))]["oracle_expert"].to_numpy()
        routed = route_probabilities(evaluation_prob, oracle_labels)
        frontier_rows.append(
            {
                "system": "oracle",
                "lambda": float(lam),
                **summarize_frontier_point(routed, expert_cost_ms, 0.0, expert_parameters),
            }
        )

    for lam in LAMBDA_SWEEP:
        thresholds, calibration_objective = tune_threshold_router(
            calibration_prob, calibration_rows, expert_cost_ms, float(lam)
        )
        routing = common.fragmentation_threshold_router(
            evaluation_rows[ROUTER_FEATURES], thresholds=thresholds, experts=ROUTER_EXPERTS
        )
        routed = route_probabilities(evaluation_prob, np.asarray(routing))
        overhead = measure_threshold_overhead_ms(suite, word_vocab, latency_texts, thresholds)
        frontier_rows.append(
            {
                "system": "fragmentation_threshold",
                "lambda": float(lam),
                "threshold_low": thresholds[0],
                "threshold_high": thresholds[1],
                "calibration_objective": calibration_objective,
                **summarize_frontier_point(routed, expert_cost_ms, overhead, expert_parameters),
            }
        )
        routed_systems[("fragmentation_threshold", float(lam))] = routed

    learned_models = {}
    for router_type in ROUTER_TYPES:
        learned_models[router_type] = fit_router_sweep(
            calibration_rows, calibration_oracle, router_type=router_type
        )
        for lam, estimator in learned_models[router_type].items():
            routing = predict_router(estimator, evaluation_rows)
            routed = route_probabilities(evaluation_prob, routing)
            overhead = measure_router_overhead_ms(estimator, suite, word_vocab, latency_texts)
            frontier_rows.append(
                {
                    "system": f"learned_{router_type}",
                    "lambda": float(lam),
                    **summarize_frontier_point(routed, expert_cost_ms, overhead, expert_parameters),
                }
            )
            routed_systems[(f"learned_{router_type}", float(lam))] = routed
            regret = lambda_routing_regret(routed, evaluation_oracle, expert_cost_ms, float(lam))
            regret["system"] = f"learned_{router_type}"
            regret["lambda"] = float(lam)
            regret_frames.append(regret)

    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_adaptive_frontier_{run_id}.csv"), index=False)
    pd.concat(regret_frames, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, f"{backbone}_{label}_routing_regret_{run_id}.csv"), index=False
    )
    return frontier, routed_systems, learned_models, evaluation_oracle


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    observed = abs(values.mean())
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        stats.append(abs((values * np.asarray(signs)).mean()))
    return float(np.mean(np.asarray(stats) >= observed - 1e-15))


def paired_effect_family(degradation: pd.DataFrame, fixed_name: str) -> pd.DataFrame:
    from scipy.stats import t

    rows = []
    for lam in LAMBDA_SWEEP:
        adaptive = f"learned_logistic_regression_lambda_{lam:g}"
        sub = degradation[
            degradation["system"].isin([adaptive, fixed_name])
            & degradation["corruption_level"].gt(0)
        ]
        paired = (
            sub.groupby(["system", "seed"])["relative_macro_f1_degradation"]
            .mean()
            .unstack("system")
            .dropna()
        )
        diff = paired[adaptive] - paired[fixed_name]
        n = len(diff)
        mean = float(diff.mean())
        sd = float(diff.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        crit = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
        rows.append(
            {
                "lambda": float(lam),
                "adaptive": adaptive,
                "fixed": fixed_name,
                "n_seeds": n,
                "mean_paired_difference": mean,
                "ci95_low": mean - crit * se if n > 1 else np.nan,
                "ci95_high": mean + crit * se if n > 1 else np.nan,
                "cohens_dz": mean / sd if n > 1 and sd > 0 else np.nan,
                "p_value_signflip": exact_signflip_p(diff.to_numpy()),
            }
        )
    frame = pd.DataFrame(rows)
    valid = frame["p_value_signflip"].notna()
    adjusted = np.full(len(frame), np.nan)
    if valid.any():
        adjusted[valid.to_numpy()] = multipletests(frame.loc[valid, "p_value_signflip"], method="holm")[1]
    frame["p_value_holm"] = adjusted
    frame["significant_holm_0.05"] = frame["p_value_holm"].lt(0.05)
    return frame


def measure_inference_ram(backbone, suite, word_vocab, test_df, run_id, primary_router, primary_lambda):
    texts = test_df["Description"].iloc[:64].to_numpy()
    rows = []
    for seed in TRAINING_SEEDS:
        for expert in ROUTER_EXPERTS:
            monitor = PeakRAMMonitor()
            monitor.start()
            try:
                model = keras.models.load_model(model_path(backbone, expert, seed, run_id))
                x = suite.vectorize(expert, texts)
                _ = model.predict(x, batch_size=1, verbose=0)
            finally:
                stats = monitor.stop()
                if "model" in locals():
                    del model
                keras.backend.clear_session()
                gc.collect()
            rows.append({"system": f"fixed_{expert}", "seed": seed, **stats})

        monitor = PeakRAMMonitor()
        monitor.start()
        models = {}
        try:
            models = {e: keras.models.load_model(model_path(backbone, e, seed, run_id)) for e in ROUTER_EXPERTS}
            features = common.compute_instability_features(texts, suite, word_vocab)
            chosen = predict_router(primary_router, features)
            for expert in ROUTER_EXPERTS:
                mask = chosen == expert
                if mask.any():
                    x = suite.vectorize(expert, np.asarray(texts)[mask])
                    _ = models[expert].predict(x, batch_size=1, verbose=0)
        finally:
            stats = monitor.stop()
            for model in models.values():
                del model
            keras.backend.clear_session()
            gc.collect()
        rows.append(
            {
                "system": "learned_logistic_regression",
                "lambda": float(primary_lambda),
                "seed": seed,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def failure_analysis(backbone, run_id, routed_with_regret, prob_table, features):
    diagnostics = confidence_failure_diagnostics(prob_table, features)
    key = ["sample_id", "seed", "corruption_level"]
    columns = key + ["correct_word", "correct_char"] + ROUTER_FEATURES
    frame = routed_with_regret.merge(diagnostics[columns], on=key, how="left", validate="one_to_one")
    frame = frame.rename(columns={"correct": "correct_chosen"})
    labeled = label_failure_modes(frame)
    labeled.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_failure_cases_{run_id}.csv"), index=False)
    counts = (
        labeled[labeled["failure_mode"].ne("")]["failure_mode"]
        .value_counts()
        .rename_axis("failure_mode")
        .reset_index(name="count")
    )
    counts.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_failure_mode_counts_{run_id}.csv"), index=False)


def main(backbone: str):
    enable_determinism()
    ensure_dirs()
    run_id = new_run_id()
    print(f"RUN_ID={run_id} backbone={backbone}")

    train_val = load_train_val_set()
    train_df, val_df = make_train_val_split(train_val)
    test_df = load_test_set()
    calibration_df = load_router_calibration_set()
    save_split_ids(train_df, val_df, run_id)

    suite = build_tokenization_suite(train_df["Description"].to_numpy())
    word_vocab = make_word_vocab_index(suite)
    save_artifacts(suite, run_id)

    test_sets, _ = common.build_nested_test_sets(test_df, CORRUPTION_LEVELS, 0)
    calibration_sets, _ = build_router_calibration_sets(calibration_df)
    ood_calibration_sets, _ = build_router_calibration_sets(calibration_df, holdout_family=OOD_HOLDOUT_FAMILY)
    ood_test_sets, _ = build_true_ood_sets(test_df, OOD_HOLDOUT_FAMILY)

    # Frozen six-tokenizer expert study.
    results = run_fixed_comparison(
        backbone=backbone,
        suite=suite,
        train_df=train_df,
        val_df=val_df,
        test_sets=test_sets,
        run_id=run_id,
    )
    metrics = results["metrics"]
    resources = results["resource_usage"]
    predictions = results["predictions"]

    # BPE-dropout is a required simpler baseline.
    dropout_metrics, dropout_predictions, dropout_resources = train_bpe_dropout(
        backbone, suite, train_df, val_df, test_sets, run_id
    )
    dropout_metrics.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_metrics_{run_id}.csv"), index=False)
    dropout_predictions.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_predictions_{run_id}.csv"), index=False)
    dropout_resources.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_resources_{run_id}.csv"), index=False)

    # Predictions for the disjoint router-calibration set and true OOD protocol.
    _, calibration_predictions = evaluate_saved(backbone, suite, calibration_sets, run_id)
    _, ood_calibration_predictions = evaluate_saved(backbone, suite, ood_calibration_sets, run_id)
    _, ood_predictions = evaluate_saved(backbone, suite, ood_test_sets, run_id)

    calibration_prob = build_expert_probability_table(calibration_predictions, experts=ROUTER_EXPERTS)
    test_prob = build_expert_probability_table(predictions, experts=ROUTER_EXPERTS)
    ood_calibration_prob = build_expert_probability_table(ood_calibration_predictions, experts=ROUTER_EXPERTS)
    ood_test_prob = build_expert_probability_table(ood_predictions, experts=ROUTER_EXPERTS)

    calibration_features = instability_frame(calibration_sets, suite, word_vocab)
    test_features = instability_frame(test_sets, suite, word_vocab)
    ood_calibration_features = instability_frame(ood_calibration_sets, suite, word_vocab)
    ood_test_features = instability_frame(ood_test_sets, suite, word_vocab)
    test_features.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_instability_features_{run_id}.csv"), index=False)

    # H1 analysis: error, confidence degradation, coarse-vs-fine loss gap, plus
    # clean-counterpart delta diagnostics. The delta columns are never router inputs.
    diagnostics = confidence_failure_diagnostics(test_prob, test_features)
    correlations = diagnostic_correlations(diagnostics)
    diagnostics.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{run_id}_failure_diagnostics.csv"), index=False)
    correlations.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{run_id}_diagnostic_correlations.csv"), index=False)

    # End-to-end fixed expert CPU cost includes tokenization/vectorization.
    latency, fixed_costs = measure_fixed_costs(
        backbone, suite, test_df, run_id, [*ROUTER_EXPERTS, "bpe_5000"]
    )
    latency.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_latency_{run_id}.csv"), index=False)
    expert_costs = {e: fixed_costs[e] for e in ROUTER_EXPERTS}
    params = expert_parameter_map(resources, ROUTER_EXPERTS)

    id_frontier, id_routed, id_models, id_oracle = fit_routing_family(
        backbone, run_id, "id", suite, word_vocab,
        calibration_prob, calibration_features,
        test_prob, test_features,
        expert_costs, params,
        test_df["Description"].to_numpy(),
    )
    ood_frontier, _, _, _ = fit_routing_family(
        backbone, run_id, "ood", suite, word_vocab,
        ood_calibration_prob, ood_calibration_features,
        ood_test_prob, ood_test_features,
        expert_costs, params,
        test_df["Description"].to_numpy(),
    )
    ood_frontier.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_ood_routing_frontier_{run_id}.csv"), index=False)

    # Add fixed systems and BPE dropout to the same measured CPU frontier.
    frontier_rows = []
    all_params = expert_parameter_map(resources, FIXED_FRONTIER_MODELS)
    for name in FIXED_FRONTIER_MODELS:
        routed = fixed_routed(predictions, name)
        frontier_rows.append(
            {
                "system": f"fixed_{name}",
                "lambda": np.nan,
                **summarize_frontier_point(routed, fixed_costs, 0.0, all_params),
            }
        )

    dropout_latency, dropout_cost = measure_fixed_costs(
        backbone, suite, test_df, run_id, ["bpe_500_dropout"]
    )
    dropout_latency.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_latency_{run_id}.csv"), index=False)
    dropout_routed = predictions_to_single_system_routed(
        dropout_predictions, "bpe_500_dropout", BPE_SEQUENCE_LENGTHS[500]
    )
    dropout_params = {
        "bpe_500_dropout": int(dropout_resources["n_parameters"].median())
    }
    frontier_rows.append(
        {
            "system": "bpe_500_dropout",
            "lambda": np.nan,
            **summarize_frontier_point(dropout_routed, dropout_cost, 0.0, dropout_params),
        }
    )
    frontier = pd.concat([pd.DataFrame(frontier_rows), id_frontier], ignore_index=True)
    frontier.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_routing_frontier_{run_id}.csv"), index=False)

    # Per-seed/per-level clean-relative degradation for fixed, dropout and all
    # logistic lambda policies.
    degradation_frames = []
    systems = {f"fixed_{m}": fixed_routed(predictions, m) for m in FIXED_FRONTIER_MODELS}
    systems["bpe_500_dropout"] = dropout_routed
    for (system, lam), routed in id_routed.items():
        if system == "learned_logistic_regression":
            systems[f"learned_logistic_regression_lambda_{lam:g}"] = routed
    for name, routed in systems.items():
        frame = relative_f1_degradation(routed)
        frame.insert(0, "system", name)
        degradation_frames.append(frame)
    degradation = pd.concat(degradation_frames, ignore_index=True)
    degradation.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_relative_degradation_{run_id}.csv"), index=False)

    # Predefined statistical family: every learned-logistic lambda vs the fixed
    # representation with the highest aggregate F1. Holm correct across lambdas.
    fixed_rows = frontier[frontier["system"].str.startswith("fixed_")]
    best_fixed = fixed_rows.loc[fixed_rows["macro_f1"].idxmax(), "system"]
    effects = paired_effect_family(degradation, best_fixed)
    effects.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_paired_effects_{run_id}.csv"), index=False)

    # McNemar/Holm uses identical final rows/seeds; no seed split is used for router fitting.
    mid_lambda = float(LAMBDA_SWEEP[len(LAMBDA_SWEEP) // 2])
    primary = id_routed[("learned_logistic_regression", mid_lambda)].copy()
    primary["system"] = "learned_logistic_regression"
    prediction_frames = []
    for expert in ROUTER_EXPERTS:
        frame = fixed_routed(predictions, expert).copy()
        frame["system"] = expert
        prediction_frames.append(frame)
    prediction_frames.append(primary)
    mcnemar_holm_table(
        pd.concat(prediction_frames, ignore_index=True),
        system_order=[*ROUTER_EXPERTS, "learned_logistic_regression"],
        seeds=TRAINING_SEEDS,
        corruption_levels=CORRUPTION_LEVELS,
        csv_name=f"{backbone}_mcnemar_systems_{run_id}.csv",
    )

    primary_regret = lambda_routing_regret(primary, id_oracle, expert_costs, mid_lambda)
    failure_analysis(backbone, run_id, primary_regret, test_prob, test_features)

    # Resident versus active cost and peak inference RAM.
    resident_rows = []
    for seed in TRAINING_SEEDS:
        paths = {e: model_path(backbone, e, seed, run_id) for e in ROUTER_EXPERTS}
        resident_rows.append(
            {
                "seed": seed,
                "resident_experts_mb": resident_system_size_mb(paths, id_models[PRIMARY_ROUTER][mid_lambda]),
                "router_lambda": mid_lambda,
            }
        )
    pd.DataFrame(resident_rows).to_csv(os.path.join(RESULTS_DIR, f"{backbone}_resident_cost_{run_id}.csv"), index=False)

    inference_ram = measure_inference_ram(
        backbone, suite, word_vocab, test_df, run_id, id_models[PRIMARY_ROUTER][mid_lambda], mid_lambda
    )
    inference_ram.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_inference_ram_{run_id}.csv"), index=False)

    manifest = {
        "run_id": run_id,
        "backbone": backbone,
        "protocol": "v2_authoritative_leakage_safe",
        "router_calibration_source": "next 500 unused AG News train rows per class after frozen v1 pool",
        "router_calibration_size": len(calibration_df),
        "final_test_used_for_router_fitting": False,
        "ood_holdout_family": OOD_HOLDOUT_FAMILY,
        "ood_family_excluded_from_ood_router_calibration": True,
        "lambda_sweep": [float(x) for x in LAMBDA_SWEEP],
        "router_types": ROUTER_TYPES,
        "primary_router": PRIMARY_ROUTER,
        "primary_lambda": mid_lambda,
        "fixed_frontier_models": FIXED_FRONTIER_MODELS,
        "bpe_dropout_on_frontier": True,
        "frontier_x_axis": "mean_total_cpu_latency_ms",
        "expert_latency_includes_tokenization": True,
        "adaptive_latency_includes_feature_and_router_overhead": True,
        "clean_counterpart_deltas_are_router_inputs": False,
        "statistical_family": "learned logistic lambdas vs best fixed representation, Holm corrected",
    }
    save_json(manifest, os.path.join(RESULTS_DIR, f"run_manifest_{backbone}_{run_id}.json"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn", "tcn"], required=True)
    args = parser.parse_args()
    main(args.backbone)
