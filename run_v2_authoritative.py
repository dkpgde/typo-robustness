"""Authoritative runner for Typo Robustness v2.

Usage
-----
python run_v2_authoritative.py --backbone cnn
python run_v2_authoritative.py --backbone tcn

The script deliberately performs the same protocol for both backbones. It is
intended to supersede the exploratory v2 notebooks for final reported results.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict

import keras
import numpy as np
import pandas as pd

from common import (
    ARTIFACTS_DIR,
    BPE_DROPOUT_PROB,
    BPE_SEQUENCE_LENGTHS,
    CORRUPTION_LEVELS,
    FIGURES_DIR,
    LAMBDA_SWEEP,
    MODEL_ORDER,
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
    config_snapshot,
    enable_determinism,
    ensure_dirs,
    evaluate_model,
    fit_learned_router,
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
    add_clean_counterpart_deltas,
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
    paired_seed_effects,
    predict_router,
    relative_f1_degradation,
    resident_system_size_mb,
    router_feature_rows,
    summarize_frontier_point,
    tune_threshold_router,
)

FIXED_FRONTIER_MODELS = ["word", "bpe_500", "bpe_5000", "char"]
ROUTER_TYPES = ["logistic_regression", "decision_tree", "gradient_boosting"]


def model_path(backbone: str, model: str, seed: int, run_id: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "models", f"{backbone}_{model}_seed{seed}_{run_id}.keras")


def builder(backbone: str, sequence_length: int, vocab_size: int):
    if backbone == "cnn":
        return lambda: build_cnn(sequence_length, vocab_size, 32)
    return lambda: build_tcn(sequence_length, vocab_size)


def evaluate_saved_models(backbone, suite, datasets, run_id, models=ROUTER_EXPERTS):
    metrics, predictions = [], []
    for model_name in models:
        for seed in TRAINING_SEEDS:
            path = model_path(backbone, model_name, seed, run_id)
            model = keras.models.load_model(path)
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
    routed = route_probabilities(table, np.asarray([model_name] * len(table)))
    return routed


def measure_expert_costs(backbone, suite, test_df, run_id):
    calibration_texts = test_df["Description"].iloc[:64].to_numpy()
    rows = []
    for expert in ROUTER_EXPERTS:
        for seed in TRAINING_SEEDS:
            path = model_path(backbone, expert, seed, run_id)
            model = keras.models.load_model(path)
            ms = measure_end_to_end_expert_latency_ms(model, suite, expert, calibration_texts)
            rows.append({"model": expert, "seed": seed, "end_to_end_latency_ms": ms})
            del model
            keras.backend.clear_session()
            gc.collect()
    df = pd.DataFrame(rows)
    return df, df.groupby("model")["end_to_end_latency_ms"].mean().to_dict()


def expert_parameter_map(resource_usage: pd.DataFrame) -> dict[str, int]:
    return (
        resource_usage[resource_usage["model"].isin(ROUTER_EXPERTS)]
        .groupby("model")["n_parameters"]
        .median()
        .astype(int)
        .to_dict()
    )


def train_bpe_dropout(backbone, suite, train_df, val_df, test_sets, run_id):
    dropout_tokenizer = make_bpe_dropout_tokenizer(suite.bpe_tokenizers[500], dropout=BPE_DROPOUT_PROB)
    train_y = train_df["Class Index"].to_numpy() - 1
    val_y = val_df["Class Index"].to_numpy() - 1
    val_x = suite.vectorize("bpe_500", val_df["Description"].to_numpy())
    metrics, predictions, resources = [], [], []
    for seed in TRAINING_SEEDS:
        sequence = BPEDropoutTrainingSequence(train_df["Description"].to_numpy(), train_y, dropout_tokenizer)
        model, history, stats = train_model(
            model_builder=builder(backbone, BPE_SEQUENCE_LENGTHS[500], suite.bpe_tokenizers[500].get_vocab_size()),
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
        out_path = model_path(backbone, "bpe_500_dropout", seed, run_id)
        model.save(out_path)
        del model
        keras.backend.clear_session()
        gc.collect()
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(resources)


def primary_diagnostics(prob_table, features, prefix):
    diagnostics = confidence_failure_diagnostics(prob_table, features)
    correlations = diagnostic_correlations(diagnostics)
    diagnostics.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_failure_diagnostics.csv"), index=False)
    correlations.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_diagnostic_correlations.csv"), index=False)
    return diagnostics, correlations


def fit_and_evaluate_routing(
    backbone,
    run_id,
    suite,
    word_vocab,
    calibration_prob,
    calibration_features,
    test_prob,
    test_features,
    expert_cost_ms,
    expert_parameters,
    test_texts,
    label,
):
    cal_rows = router_feature_rows(calibration_prob, calibration_features)
    test_rows = router_feature_rows(test_prob, test_features)
    cal_oracle = oracle_targets(calibration_prob, expert_cost_ms)
    test_oracle = oracle_targets(test_prob, expert_cost_ms)
    cal_oracle.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_calibration_oracle_{run_id}.csv"), index=False)
    test_oracle.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_test_oracle_{run_id}.csv"), index=False)

    frontier_rows = []
    routed_systems = {}
    regrets = []

    # Oracle curve on final rows.
    for lam in LAMBDA_SWEEP:
        oracle_labels = test_oracle[test_oracle["lambda"].eq(float(lam))]["oracle_expert"].to_numpy()
        routed = route_probabilities(test_prob, oracle_labels)
        summary = summarize_frontier_point(routed, expert_cost_ms, 0.0, expert_parameters)
        frontier_rows.append({"system": "oracle", "lambda": float(lam), **summary})

    # Threshold policy is tuned only on calibration examples for each lambda.
    for lam in LAMBDA_SWEEP:
        thresholds, calibration_objective = tune_threshold_router(
            calibration_prob, cal_rows, expert_cost_ms, float(lam)
        )
        routing = __import__("common").fragmentation_threshold_router(
            test_rows[ROUTER_FEATURES], thresholds=thresholds, experts=ROUTER_EXPERTS
        )
        routed = route_probabilities(test_prob, np.asarray(routing))
        overhead = measure_threshold_overhead_ms(suite, word_vocab, test_texts, thresholds)
        summary = summarize_frontier_point(routed, expert_cost_ms, overhead, expert_parameters)
        frontier_rows.append(
            {
                "system": "fragmentation_threshold",
                "lambda": float(lam),
                "threshold_low": thresholds[0],
                "threshold_high": thresholds[1],
                "calibration_objective": calibration_objective,
                **summary,
            }
        )
        routed_systems[("fragmentation_threshold", float(lam))] = routed

    # Learned policies: a distinct model for each lambda and router family.
    learned = {}
    for router_type in ROUTER_TYPES:
        learned[router_type] = fit_router_sweep(cal_rows, cal_oracle, router_type=router_type)
        for lam, estimator in learned[router_type].items():
            routing = predict_router(estimator, test_rows)
            routed = route_probabilities(test_prob, routing)
            overhead = measure_router_overhead_ms(estimator, suite, word_vocab, test_texts)
            summary = summarize_frontier_point(routed, expert_cost_ms, overhead, expert_parameters)
            frontier_rows.append(
                {"system": f"learned_{router_type}", "lambda": float(lam), **summary}
            )
            routed_systems[(f"learned_{router_type}", float(lam))] = routed
            regret = lambda_routing_regret(routed, test_oracle, expert_cost_ms, float(lam))
            regret["system"] = f"learned_{router_type}"
            regret["lambda"] = float(lam)
            regrets.append(regret)

    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_{label}_routing_frontier_{run_id}.csv"), index=False)
    pd.concat(regrets, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, f"{backbone}_{label}_routing_regret_{run_id}.csv"), index=False
    )
    return frontier, routed_systems, learned, test_oracle


def add_fixed_frontier(frontier, predictions_long, expert_cost_ms, expert_parameters):
    rows = []
    for name in FIXED_FRONTIER_MODELS:
        routed = fixed_routed(predictions_long, name)
        # bpe_5000 is not a router expert, so use its separately measured sequence
        # and latency later if available; here CPU cost is filled by caller if absent.
        if name in expert_cost_ms:
            cost_map = expert_cost_ms
        else:
            cost_map = {**expert_cost_ms, name: np.nan}
        rows.append(
            {
                "system": f"fixed_{name}",
                "lambda": np.nan,
                **summarize_frontier_point(routed, cost_map, 0.0, expert_parameters),
            }
        )
    return pd.concat([pd.DataFrame(rows), frontier], ignore_index=True)


def failure_analysis(backbone, run_id, routed, prob_table, features):
    # Build the exact fields expected by common.label_failure_modes.
    diagnostics = confidence_failure_diagnostics(prob_table, features)
    key = ["sample_id", "seed", "corruption_level"]
    merged = routed.merge(
        diagnostics[key + ["correct_word", "correct_char"] + ROUTER_FEATURES],
        on=key,
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"correct": "correct_chosen"})
    # regret is optional for labels; when absent, keep zero so cases still classify.
    if "regret" not in merged:
        merged["regret"] = 0.0
    labeled = label_failure_modes(merged)
    labeled.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_failure_cases_{run_id}.csv"), index=False)
    counts = (
        labeled[labeled["failure_mode"].ne("")]["failure_mode"]
        .value_counts()
        .rename_axis("failure_mode")
        .reset_index(name="count")
    )
    counts.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_failure_mode_counts_{run_id}.csv"), index=False)
    return counts


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

    test_sets, _ = __import__("common").build_nested_test_sets(test_df, CORRUPTION_LEVELS, 0)
    calibration_sets, _ = build_router_calibration_sets(calibration_df)
    ood_calibration_sets, _ = build_router_calibration_sets(calibration_df, holdout_family=OOD_HOLDOUT_FAMILY)
    ood_test_sets, _ = build_true_ood_sets(test_df, OOD_HOLDOUT_FAMILY)

    # Frozen six-tokenizer experiment.
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

    # Router-calibration and OOD predictions are produced by the already-trained
    # experts. No router-fitting row comes from the final test set.
    _, calibration_predictions = evaluate_saved_models(backbone, suite, calibration_sets, run_id)
    _, ood_calibration_predictions = evaluate_saved_models(backbone, suite, ood_calibration_sets, run_id)
    _, ood_predictions = evaluate_saved_models(backbone, suite, ood_test_sets, run_id)

    calibration_prob = build_expert_probability_table(calibration_predictions, experts=ROUTER_EXPERTS)
    test_prob = build_expert_probability_table(predictions, experts=ROUTER_EXPERTS)
    ood_calibration_prob = build_expert_probability_table(ood_calibration_predictions, experts=ROUTER_EXPERTS)
    ood_test_prob = build_expert_probability_table(ood_predictions, experts=ROUTER_EXPERTS)

    calibration_features = instability_frame(calibration_sets, suite, word_vocab)
    test_features = instability_frame(test_sets, suite, word_vocab)
    ood_calibration_features = instability_frame(ood_calibration_sets, suite, word_vocab)
    ood_test_features = instability_frame(ood_test_sets, suite, word_vocab)
    test_features.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_instability_features_{run_id}.csv"), index=False)

    diagnostics, correlations = primary_diagnostics(test_prob, test_features, f"{backbone}_{run_id}")

    latency_df, expert_cost_ms = measure_expert_costs(backbone, suite, test_df, run_id)
    latency_df.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_latency_{run_id}.csv"), index=False)
    params = expert_parameter_map(resources)

    # Main ID routing and true held-out-family OOD routing.
    frontier, routed, learned, oracle = fit_and_evaluate_routing(
        backbone, run_id, suite, word_vocab,
        calibration_prob, calibration_features,
        test_prob, test_features,
        expert_cost_ms, params,
        test_df["Description"].to_numpy(),
        "id",
    )
    ood_frontier, ood_routed, ood_learned, _ = fit_and_evaluate_routing(
        backbone, run_id, suite, word_vocab,
        ood_calibration_prob, ood_calibration_features,
        ood_test_prob, ood_test_features,
        expert_cost_ms, params,
        test_df["Description"].to_numpy(),
        "ood",
    )

    # Add fixed BPE-5000 with a directly measured end-to-end CPU cost.
    bpe5000_costs = []
    for seed in TRAINING_SEEDS:
        model = keras.models.load_model(model_path(backbone, "bpe_5000", seed, run_id))
        bpe5000_costs.append(
            measure_end_to_end_expert_latency_ms(
                model, suite, "bpe_5000", test_df["Description"].iloc[:64].to_numpy()
            )
        )
        del model
        keras.backend.clear_session()
    all_costs = {**expert_cost_ms, "bpe_5000": float(np.mean(bpe5000_costs))}
    all_params = {
        **params,
        "bpe_5000": int(resources[resources["model"].eq("bpe_5000")]["n_parameters"].median()),
    }
    frontier = add_fixed_frontier(frontier, predictions, all_costs, all_params)
    frontier.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_routing_frontier_{run_id}.csv"), index=False)

    # BPE-dropout baseline is trained/evaluated under the same backbone.
    dropout_metrics, dropout_predictions, dropout_resources = train_bpe_dropout(
        backbone, suite, train_df, val_df, test_sets, run_id
    )
    dropout_metrics.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_metrics_{run_id}.csv"), index=False)
    dropout_predictions.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_predictions_{run_id}.csv"), index=False)
    dropout_resources.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_bpe_dropout_resources_{run_id}.csv"), index=False)

    # Per-level/seed relative degradation for all final systems.
    degradation_frames = []
    fixed_systems = {}
    for name in FIXED_FRONTIER_MODELS:
        fixed_systems[f"fixed_{name}"] = fixed_routed(predictions, name)
    # choose the predeclared primary learned family, but keep every lambda.
    all_eval_systems = {**fixed_systems, **{f"learned_logistic_regression_lambda_{lam:g}": r for (s, lam), r in routed.items() if s == "learned_logistic_regression"}}
    for system, frame in all_eval_systems.items():
        d = relative_f1_degradation(frame)
        d.insert(0, "system", system)
        degradation_frames.append(d)
    degradation = pd.concat(degradation_frames, ignore_index=True)
    degradation.to_csv(os.path.join(RESULTS_DIR, f"{backbone}_relative_degradation_{run_id}.csv"), index=False)

    # Statistical comparison family: adaptive logistic policies vs best fixed
    # representation, selected by the final CPU/F1 frontier rather than by name.
    fixed_frontier = frontier[frontier["system"].str.startswith("fixed_")]
    best_fixed_name = fixed_frontier.loc[fixed_frontier["macro_f1"].idxmax(), "system"]
    effects = []
    for lam in LAMBDA_SWEEP:
        adaptive_name = f"learned_logistic_regression_lambda_{lam:g}"
        effects.append(paired_seed_effects(degradation, adaptive_name, best_fixed_name))
    pd.DataFrame(effects).to_csv(os.path.join(RESULTS_DIR, f"{backbone}_paired_effects_{run_id}.csv"), index=False)

    # McNemar/Holm on a prespecified representative lambda (middle of sweep),
    # with all systems evaluated on identical final rows and all nine seeds.
    mid_lambda = float(LAMBDA_SWEEP[len(LAMBDA_SWEEP) // 2])
    primary = routed[("learned_logistic_regression", mid_lambda)].copy()
    primary["system"] = "learned_logistic_regression"
    systems_predictions = []
    for name in ROUTER_EXPERTS:
        f = fixed_systems[f"fixed_{name}"].copy()
        f["system"] = name
        systems_predictions.append(f)
    systems_predictions.append(primary)
    mcnemar_holm_table(
        pd.concat(systems_predictions, ignore_index=True),
        system_order=[*ROUTER_EXPERTS, "learned_logistic_regression"],
        seeds=TRAINING_SEEDS,
        corruption_levels=CORRUPTION_LEVELS,
        csv_name=f"{backbone}_mcnemar_systems_{run_id}.csv",
    )

    # Failure analysis for both backbones, using the primary policy.
    primary_regret = lambda_routing_regret(primary, oracle, expert_cost_ms, mid_lambda)
    failure_analysis(backbone, run_id, primary_regret, test_prob, test_features)

    # Resident/active cost accounting. Resident adaptive cost stores all three
    # experts; active params are already in the frontier summaries.
    resident_rows = []
    for seed in TRAINING_SEEDS:
        paths = {e: model_path(backbone, e, seed, run_id) for e in ROUTER_EXPERTS}
        resident_rows.append(
            {
                "seed": seed,
                "resident_experts_mb": resident_system_size_mb(paths),
                "peak_training_ram_mb_max_expert": float(
                    resources[(resources["seed"].eq(seed)) & (resources["model"].isin(ROUTER_EXPERTS))]["peak_ram_mb"].max()
                ),
            }
        )
    pd.DataFrame(resident_rows).to_csv(os.path.join(RESULTS_DIR, f"{backbone}_resident_cost_{run_id}.csv"), index=False)

    manifest = {
        "run_id": run_id,
        "backbone": backbone,
        "protocol": "v2_authoritative_leakage_safe",
        "router_calibration_rows": len(calibration_df),
        "router_calibration_source": "unused AG News training rows after first 3500/class",
        "ood_holdout_family": OOD_HOLDOUT_FAMILY,
        "lambda_sweep": LAMBDA_SWEEP,
        "router_types": ROUTER_TYPES,
        "fixed_frontier_models": FIXED_FRONTIER_MODELS,
        "primary_router": "logistic_regression",
        "primary_lambda": mid_lambda,
        "diagnostic_targets": ["word_error", "confidence_degradation_word", "loss_gap_word_minus_char"],
        "frontier_x_axis": "mean_total_cpu_latency_ms",
        "clean_counterpart_deltas_are_router_inputs": False,
    }
    save_json(manifest, os.path.join(RESULTS_DIR, f"run_manifest_{backbone}_{run_id}.json"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn", "tcn"], required=True)
    args = parser.parse_args()
    main(args.backbone)
