"""Measure held-out CT-SGF evidence coverage and full-horizon calibration.

The input JSON object has horizon, bandwidths, bank, and evaluation fields.
Each rollout has group, features, observed_steps, and first_event. Use one
consistent log or location unit for group, and keep bank and evaluation groups
disjoint. Features are chronological rows from the candidate and inference-time
context. first_event is a zero-based step index or null. observed_steps counts
an observed prefix, including the event step when an event is observed. All
rollouts use the same interval and horizon. Feature rows may extend beyond the
observed prefix when a complete candidate can still be queried.

The output reports step support on supplied features and on observed at-risk
steps. A known horizon label requires a recorded event or complete observation.
Both calibration scores use the same known-label rollouts with supported queries
at every step of the horizon. Scores are null when that subset is empty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ct_sgf import HazardModel, Rollout
from ct_sgf_baselines import EventOnlyModel, StepwiseLogistic, TrajectoryLogistic, TrajectoryLogKDE


def read_rollouts(records: list, horizon: int, section: str) -> list[Rollout]:
    if not isinstance(records, list):
        raise ValueError(f"{section} must be an array")
    rollouts = []
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("expected a rollout object")
            features = record["features"]
            if not isinstance(features, list) or not 1 <= len(features) <= horizon:
                raise ValueError("features must contain between 1 and horizon rows")
            rollouts.append(Rollout(
                record["group"], features, record["observed_steps"], record["first_event"],
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{section}[{index}]: {error}") from error
    return rollouts


def coverage(supported: int, total: int) -> dict:
    return {
        "supported": supported,
        "total": total,
        "fraction": supported / total if total else None,
    }


def calibration(forecasts) -> dict:
    """Score probability, log survival and observed failure, with optional log failure."""
    bins = [[] for _ in range(10)]
    brier_terms = []
    log_losses = []
    for probability, log_survival, failed, *log_failure in forecasts:
        bins[min(int(probability * 10), 9)].append((probability, failed))
        brier_terms.append((probability - failed) ** 2)
        if failed:
            log_losses.append(-log_failure[0] if log_failure else -math.log(probability))
        else:
            log_losses.append(-log_survival)

    count = len(forecasts)
    return {
        "count": count,
        "brier_score": math.fsum(brier_terms) / count if count else None,
        "negative_log_likelihood": math.fsum(log_losses) / count if count else None,
        "failure_probability_bins": [
            {
                "lower_inclusive": index / 10,
                "upper": (index + 1) / 10,
                "upper_inclusive": index == 9,
                "count": len(bucket),
                "mean_predicted_failure": (
                    math.fsum(probability for probability, _ in bucket) / len(bucket)
                    if bucket else None
                ),
                "observed_failure_rate": (
                    sum(failed for _, failed in bucket) / len(bucket) if bucket else None
                ),
            }
            for index, bucket in enumerate(bins)
        ],
    }


def evaluate(model: HazardModel, evaluation: list[Rollout], horizon: int, *, baselines=None) -> dict:
    supplied_steps = supported_steps = at_risk_steps = supported_at_risk = 0
    known_labels = full_horizon_queries = labeled_full_horizon = supported_full_horizon = 0
    forecasts = []
    shared_forecasts = None if baselines is None else {name: [] for name in ("conditional", *baselines)}
    for rollout in evaluation:
        forecast = model.predict(rollout.features)
        num_steps = len(forecast.steps)
        at_risk = rollout.observed_steps if rollout.first_event is None else rollout.first_event + 1
        supplied_steps += num_steps
        supported_steps += sum(step.supported for step in forecast.steps)
        at_risk_steps += at_risk
        supported_at_risk += sum(step.supported for step in forecast.steps[:at_risk])

        failed = rollout.first_event is not None
        label_known = failed or rollout.observed_steps == horizon
        known_labels += label_known
        if num_steps < horizon:
            continue
        full_horizon_queries += 1
        supported_full_horizon += forecast.supported
        if not label_known:
            continue
        labeled_full_horizon += 1
        if forecast.supported:
            example = (forecast.failure_probability, forecast.log_survival, failed)
            forecasts.append(example)
            if shared_forecasts is None or any(baseline is None for baseline in baselines.values()):
                continue
            alternatives = {name: scorer.predict(rollout.features) for name, scorer in baselines.items()}
            if not all(getattr(prediction, "supported", True) for prediction in alternatives.values()):
                continue
            shared_forecasts["conditional"].append(example)
            for name, prediction in alternatives.items():
                alternative = (prediction.failure_probability, prediction.log_survival, failed)
                if hasattr(prediction, "log_failure"):
                    alternative = (*alternative, prediction.log_failure)
                shared_forecasts[name].append(alternative)

    prior_hazard = model.alpha / (model.alpha + model.beta)
    prior_log_survival = horizon * math.log1p(-prior_hazard)
    prior_probability = -math.expm1(prior_log_survival)
    prior_forecasts = [
        (prior_probability, prior_log_survival, failed) for _, _, failed in forecasts
    ]
    report = {
        "step_support": {
            "all_supplied_queries": coverage(supported_steps, supplied_steps),
            "observed_at_risk_only": coverage(supported_at_risk, at_risk_steps),
            "missing_query_steps": horizon * len(evaluation) - supplied_steps,
        },
        "rollouts": {
            "total": len(evaluation),
            "known_full_horizon_labels": known_labels,
            "unknown_due_to_right_censoring": len(evaluation) - known_labels,
            "full_horizon_queries": full_horizon_queries,
            "supported_full_horizon_queries": supported_full_horizon,
            "known_labels_with_full_horizon_queries": labeled_full_horizon,
            "known_labels_with_supported_full_horizon_queries": len(forecasts),
        },
        "conditional_hazard": calibration(forecasts),
        "prior_only": calibration(prior_forecasts),
    }
    if shared_forecasts is not None:
        report["common_supported_calibration"] = {
            name: calibration(items) for name, items in shared_forecasts.items()
        }
        report["available_baselines"] = [
            name for name, scorer in baselines.items() if scorer is not None
        ]
    return report


def run_input(
    data: dict, bank_sizes: list[int] | None = None, *,
    compare_baselines=False, event_only_features=None,
) -> dict:
    if not isinstance(data, dict):
        raise ValueError("input must be a JSON object")
    horizon = data["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    bandwidths = data["bandwidths"]
    if not isinstance(bandwidths, list) or not bandwidths:
        raise ValueError("bandwidths must be a nonempty array")
    progress_gate = data.get("progress_gate")
    if compare_baselines and event_only_features is None:
        event_only_features = data.get("event_only_features")
    bank = read_rollouts(data["bank"], horizon, "bank")
    evaluation = read_rollouts(data["evaluation"], horizon, "evaluation")
    bank_groups = {rollout.group for rollout in bank}
    eval_groups = {rollout.group for rollout in evaluation}
    overlap = sorted(bank_groups & eval_groups)
    if overlap:
        raise ValueError(f"bank and evaluation group IDs overlap: {', '.join(overlap[:5])}")
    if not bank_groups:
        raise ValueError("bank must contain at least one rollout")
    if bank_sizes is not None and any(not 1 <= size <= len(bank_groups) for size in bank_sizes):
        raise ValueError("each bank size must count 1 to all available bank groups")

    reports = []
    ordered_groups = sorted(bank_groups)
    sizes = [len(bank_groups)] if bank_sizes is None else sorted(set(bank_sizes))
    for size in sizes:
        selected = set(ordered_groups[:size])
        selected_bank = [rollout for rollout in bank if rollout.group in selected]
        model = HazardModel(selected_bank, bandwidths, progress_gate=progress_gate)
        baselines = None
        if compare_baselines or event_only_features is not None:
            baselines = {
                "log_kde": TrajectoryLogKDE.fit(selected_bank, bandwidths, horizon),
                "trajectory_scorer": TrajectoryLogistic.fit(selected_bank, bandwidths, horizon),
                "stepwise_scorer": StepwiseLogistic.fit(selected_bank, bandwidths, horizon),
            }
            if event_only_features is not None:
                baselines["event_only"] = EventOnlyModel(selected_bank, bandwidths, event_only_features)
        reports.append({
            "bank_groups": size,
            "bank_rollouts": len(selected_bank),
            **evaluate(model, evaluation, horizon, baselines=baselines),
        })
    return {
        "horizon_steps": horizon,
        "evaluation_groups": len(eval_groups),
        "bank_subsets": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON evidence bank and held-out queries")
    parser.add_argument(
        "--bank-sizes", type=int, nargs="+", metavar="N",
        help="compare nested subsets of N groups in sorted group ID order (default: all groups)",
    )
    parser.add_argument("--compare-baselines", action="store_true", help="fit log-KDE and two logistic scorers")
    parser.add_argument(
        "--event-only-features", type=int, nargs="+", metavar="INDEX",
        help="also fit an event-only model on these feature indices",
    )
    args = parser.parse_args()
    try:
        with args.input.open(encoding="utf-8-sig") as stream:
            report = run_input(
                json.load(stream), args.bank_sizes, compare_baselines=args.compare_baselines,
                event_only_features=args.event_only_features,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
