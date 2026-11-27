"""Run a single-stage CT-SGF pilot on trusted NAVSIM v2 metric caches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import random
from statistics import fmean

from ct_sgf import HazardModel, Rollout, correct_plan
from ct_sgf_baselines import (
    EventOnlyModel, RawKernelTrajectory, StepwiseLogistic, TrajectoryLogistic, TrajectoryLogKDE,
)
from ct_sgf_evaluate import calibration
from ct_sgf_navsim_adapter import InitialCollisionError


NAVSIM_COMMIT = "0a380a9063d7162ec93d0f51e9990ebac585f720"
CONTROL_SCALE = (4.0, 1.0, 0.3)
PILOT_ACCELERATIONS = (-2.0, -1.25, -0.75, -0.4, 0.4, 0.75, 1.25)


@dataclass(frozen=True)
class Scene:
    cache: Path
    log: str
    group: str
    base: tuple | None = None
    proposals: tuple = ()


@dataclass(frozen=True)
class BankScene:
    context: tuple
    rollouts: tuple[Rollout, ...]


def parse_plan(rows) -> tuple:
    if len(rows) != 8:
        raise ValueError("each plan must have eight 0.5-second ego-relative waypoints")
    plan = tuple(tuple(float(value) for value in row) for row in rows)
    if any(len(row) != 3 or not all(math.isfinite(value) for value in row) for row in plan):
        raise ValueError("each waypoint must have finite (x, y, yaw)")
    return plan


def read_manifest(path: Path) -> tuple[list[Scene], list[Scene]]:
    def parse_scene(entry):
        cache = (path.parent / entry["cache"]).resolve()
        if not cache.is_file():
            raise ValueError(f"metric cache does not exist: {cache}")
        fields = (entry["log"], entry["group"])
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("each scene needs a nonempty log and location group")
        base = parse_plan(entry["base"]) if "base" in entry else None
        proposals = tuple(parse_plan(rows) for rows in entry.get("proposals", ()))
        if base is None and proposals:
            raise ValueError("provide a base plan with explicit proposals")
        return Scene(cache, *fields, base, proposals)

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return [parse_scene(x) for x in data["bank"]], [parse_scene(x) for x in data["evaluation"]]


def choose_cache_scenes(root: Path, scenes_per_log: int, max_logs: int, seed: int):
    by_log = {}
    for cache in sorted(root.glob("*/*/*/metric_cache.pkl")):
        log = cache.relative_to(root).parts[0]
        by_log.setdefault(log, []).append(Scene(cache, log, log))
    logs = sorted(by_log)
    if len(logs) < 3:
        raise ValueError(f"need caches from at least three logs under {root} (found {len(logs)})")
    random.Random(seed).shuffle(logs)
    logs = logs[:max_logs]
    heldout_count = max(1, len(logs) // 4)
    return tuple(
        [scene for log in subset for scene in by_log[log][:scenes_per_log]]
        for subset in (logs[heldout_count:], logs[:heldout_count])
    )


def check_split(bank: list[Scene], holdout: list[Scene]) -> None:
    if not bank or not holdout:
        raise ValueError("both the bank and evaluation need scenes")
    for split in (bank, holdout):
        if len({scene.cache.resolve() for scene in split}) != len(split):
            raise ValueError("each metric cache may appear only once in its split")
    for attribute in ("cache", "log", "group"):
        train = {getattr(scene, attribute) for scene in bank}
        test = {getattr(scene, attribute) for scene in holdout}
        if train & test:
            raise ValueError(f"bank and evaluation overlap by {attribute}")


def plans_for(adapter, scene: Scene, cache):
    if scene.base is None:
        base = parse_plan(adapter.straight_poses(cache))
        proposals = tuple(
            parse_plan(adapter.straight_poses(cache, acceleration=acceleration))
            for acceleration in PILOT_ACCELERATIONS
        )
    else:
        base, proposals = scene.base, scene.proposals
    return base, tuple(dict.fromkeys(proposal for proposal in proposals if proposal != base))


def build_bank(adapter, scenes: list[Scene], skipped: list[dict]) -> list[BankScene]:
    bank = []
    for scene in scenes:
        cache = adapter.load_cache(scene.cache)
        if cache.log_name != scene.log:
            raise ValueError(f"manifest log differs from cached log: {scene.cache}")
        context = tuple(adapter.scene_context(cache))
        base, proposals = plans_for(adapter, scene, cache)
        rollouts = []
        try:
            for plan in (base, *proposals):
                result = adapter.assess(cache, plan)
                rollouts.append(Rollout(
                    scene.group, result.features, result.observed_steps, result.first_event,
                ))
        except InitialCollisionError:
            skipped.append({"split": "bank", "scene": scene.cache.parent.name, "log": scene.log})
            continue
        bank.append(BankScene(context, tuple(rollouts)))
    return bank


def context_distance(left, right, widths) -> float:
    return math.fsum(((a - b) / width) ** 2 for a, b, width in zip(left, right, widths))


def add_context(rollout: Rollout, context: tuple) -> Rollout:
    return Rollout(
        rollout.group, tuple(context + tuple(row) for row in rollout.features),
        rollout.observed_steps, rollout.first_event,
    )


def flatten(plan):
    return tuple(value for row in plan for value in row)


def unflatten(controls):
    return tuple(tuple(controls[index:index + 3]) for index in range(0, len(controls), 3))


def rollouts_for(near):
    return [add_context(rollout, scene.context) for scene in near for rollout in scene.rollouts]


def within_pilot_bounds(candidate):
    return (
        candidate.max_abs_speed <= 35.0
        and candidate.accel_min >= -6.0
        and candidate.accel_max <= 4.0
        and candidate.max_abs_steering_rate <= 1.0
    )


def alternative(scorer, candidates, features_for, eta, min_gain, *, raw=False):
    if scorer is None or candidates[0].objective is None:
        return candidates[0]
    base = candidates[0]
    selected = base
    best = math.inf
    base_cost = None
    for entry in candidates:
        if entry.objective is None:
            continue
        features = features_for(entry.controls)
        if raw:
            risk = scorer.score(features)
        else:
            prediction = scorer.predict(features)
            if not getattr(prediction, "supported", True):
                if entry is base:
                    return base
                continue
            risk = -prediction.log_survival
        value = 0.5 * entry.distance ** 2 + eta * risk
        if entry is base:
            base_cost = value
        if value < best:
            best = value
            selected = entry
    return selected if base_cost is not None and base_cost - best > min_gain else base


def outcome(result):
    return {"first_event": result.first_event, "navsim_raw_metrics": result.score_metrics}


def summarize(rows, method, *, reference="base"):
    pairs = [
        (row["outcomes"][reference], row["outcomes"][method])
        for row in rows if reference in row["outcomes"] and method in row["outcomes"]
    ]
    if not pairs:
        return None
    base_events = [base["first_event"] is not None for base, _ in pairs]
    method_events = [result["first_event"] is not None for _, result in pairs]
    base_successes = sum(not flag for flag in base_events)
    new_failures = sum(not before and after for before, after in zip(base_events, method_events))
    averages = {}
    for metric in ("pdm_score", "ego_progress", "ego_progress_m", "history_comfort"):
        values = [(base["navsim_raw_metrics"].get(metric), other["navsim_raw_metrics"].get(metric)) for base, other in pairs]
        usable = [(base, result) for base, result in values if base is not None and result is not None]
        if usable:
            averages[metric] = {
                "base": fmean(base for base, _ in usable),
                "method": fmean(result for _, result in usable),
            }
    return {
        "paired_scenes": len(pairs),
        "reference": reference,
        "base_failure_rate": fmean(base_events),
        "method_failure_rate": fmean(method_events),
        "paired_difference": fmean(method_events) - fmean(base_events),
        "fixed_failures": sum(before and not after for before, after in zip(base_events, method_events)),
        "reference_successes": base_successes,
        "new_failures_on_reference_successes": new_failures,
        "new_failure_rate_on_reference_successes": new_failures / base_successes if base_successes else None,
        "raw_navsim_averages": averages,
    }


def evaluate_scene(adapter, scene, bank, neighbors, eta, min_gain, max_distance, evidence):
    cache = adapter.load_cache(scene.cache)
    if cache.log_name != scene.log:
        raise ValueError(f"manifest log differs from cached log: {scene.cache}")
    context = tuple(adapter.scene_context(cache))
    context_widths = tuple(adapter.context_bandwidths)
    near = sorted(bank, key=lambda item: context_distance(context, item.context, context_widths))[:neighbors]
    rollouts = rollouts_for(near)
    names = tuple(adapter.feature_names)
    offset = len(context)
    bandwidths = context_widths + tuple(adapter.bandwidths)
    progress = offset + names.index(adapter.progress_feature)
    model = HazardModel(rollouts, bandwidths, progress_gate=(progress, 12.0))
    ablated = tuple(range(offset)) + tuple(offset + names.index(name) for name in ("time_s", "entry_x_m", "entry_y_m"))
    event_only = EventOnlyModel(rollouts, bandwidths, ablated)
    scorers = {
        "event_only": event_only,
        "raw_kernel": RawKernelTrajectory.fit(rollouts, bandwidths, adapter.horizon),
        "log_kde": TrajectoryLogKDE.fit(rollouts, bandwidths, adapter.horizon),
        "trajectory_scorer": TrajectoryLogistic.fit(rollouts, bandwidths, adapter.horizon),
        "stepwise_scorer": StepwiseLogistic.fit(rollouts, bandwidths, adapter.horizon),
    }
    base, proposals = plans_for(adapter, scene, cache)
    original = flatten(base)
    scales = CONTROL_SCALE * len(base)

    @lru_cache(maxsize=None)
    def tracked(controls):
        return adapter.tracked_candidate(cache, unflatten(controls))

    def features_for(controls):
        return tuple(context + tuple(row) for row in tracked(controls).features)

    restricted = not within_pilot_bounds(tracked(original))
    allowed = []
    if not restricted:
        for plan in proposals:
            flat = flatten(plan)
            distance = math.hypot(*((a - b) / scale for a, b, scale in zip(flat, original, scales)))
            if distance <= max_distance and within_pilot_bounds(tracked(flat)):
                allowed.append(flat)
    decision = correct_plan(
        original, allowed, features_for=features_for, model=model,
        lower=(-10000.0,) * len(original), upper=(10000.0,) * len(original), control_scale=scales,
        max_distance=max_distance, eta=eta, min_gain=min_gain,
    )
    candidates = tuple(entry for entry in decision.evaluations if entry.feasible)
    selected = {"base": original, "conditional": decision.selected}
    forecast_predictions = {"conditional": decision.base_prediction}
    for name, scorer in scorers.items():
        if scorer is None:
            continue
        choice = alternative(scorer, candidates, features_for, eta, min_gain, raw=name == "raw_kernel")
        selected[name] = choice.controls
        score = scorer.predict(features_for(original))
        if getattr(score, "supported", True):
            forecast_predictions[name] = score

    # Held-out event labels enter only after every method has selected a plan.
    assessed = {entry.controls: adapter.assess(cache, unflatten(entry.controls)) for entry in candidates}
    if not restricted:
        selected["finite_oracle"] = min(
            candidates, key=lambda entry: (assessed[entry.controls].first_event is not None, entry.distance),
        ).controls
    base_event = assessed[original].first_event is not None
    if evidence is not None:
        result = assessed[original]
        evidence.append(Rollout(
            scene.group, features_for(original), result.observed_steps, result.first_event,
        ))
    forecasts = {}
    for name, prediction in forecast_predictions.items():
        if name == "raw_kernel" or not getattr(prediction, "supported", True):
            continue
        probability = prediction.failure_probability
        log_failure = prediction.log_failure if hasattr(prediction, "log_failure") else math.log(probability)
        forecasts[name] = (probability, prediction.log_survival, base_event, log_failure)
    return {
        "scene": scene.cache.parent.name, "log": scene.log, "group": scene.group,
        "searched_bank_scenes": len(near), "proposed_variants": len(proposals),
        "feasible_candidates_including_base": 0 if restricted else len(candidates),
        "supported_candidates_including_base": sum(entry.objective is not None for entry in candidates),
        "base_supported_steps": sum(step.supported for step in decision.base_prediction.steps),
        "correction_reason": "base_outside_pilot_bounds" if restricted else decision.reason,
        "outcomes": {name: outcome(assessed[controls]) for name, controls in selected.items()},
        "forecasts": forecasts,
    }


def run_pilot(
    adapter, bank_scenes, evaluation, *, neighbors=24, eta=3.0, min_gain=0.02, max_distance=6.0,
    evidence_path=None,
):
    check_split(bank_scenes, evaluation)
    skipped = []
    bank = build_bank(adapter, bank_scenes, skipped)
    if not bank:
        raise ValueError("no bank scenes remained after excluding initial collisions")
    evidence = [] if evidence_path is not None else None
    rows = []
    for scene in evaluation:
        try:
            rows.append(evaluate_scene(adapter, scene, bank, neighbors, eta, min_gain, max_distance, evidence))
        except InitialCollisionError:
            skipped.append({"split": "evaluation", "scene": scene.cache.parent.name, "log": scene.log})
    if not rows:
        raise ValueError("no evaluation scenes remained after excluding initial collisions")
    if evidence is not None:
        context_widths = tuple(adapter.context_bandwidths)
        names = tuple(adapter.feature_names)
        offset = len(context_widths)
        bundle = {
            "horizon": adapter.horizon,
            "interval_s": adapter.interval_length,
            "bandwidths": (*context_widths, *adapter.bandwidths),
            "progress_gate": (offset + names.index(adapter.progress_feature), 12.0),
            "event_only_features": (
                *range(offset), *(offset + names.index(name) for name in ("time_s", "entry_x_m", "entry_y_m")),
            ),
            "bank": [vars(rollout) for rollout in rollouts_for(bank)],
            "evaluation": [vars(rollout) for rollout in evidence],
        }
        evidence_path.write_text(json.dumps(bundle, allow_nan=False) + "\n", encoding="utf-8")
    score_names = ("conditional", "event_only", "log_kde", "trajectory_scorer", "stepwise_scorer")
    alternatives = ("event_only", "raw_kernel", "log_kde", "trajectory_scorer", "stepwise_scorer")
    shared = [row for row in rows if all(name in row["forecasts"] for name in score_names)]
    return {
        "pinned_navsim_commit": NAVSIM_COMMIT,
        "scope": "single-stage NAVSIM v2 raw score and first at-fault collision, non-reactive traffic",
        "candidate_policy": (
            "same physically feasible proposals; all evidence-based methods use conditional support; "
            "event-only may abstain on its own support; finite oracle uses all physically feasible proposals"
        ),
        "horizon_steps": adapter.horizon,
        "interval_s": adapter.interval_length,
        "bank_scenes": len(bank),
        "bank_rollouts": sum(len(scene.rollouts) for scene in bank),
        "bank_first_events": sum(
            rollout.first_event is not None for scene in bank for rollout in scene.rollouts
        ),
        "bank_groups": sorted({scene.rollouts[0].group for scene in bank}),
        "evaluation_groups": sorted({row["group"] for row in rows}),
        "skipped_initial_collisions": skipped,
        "pilot_settings": {"neighbors": neighbors, "eta": eta, "min_gain": min_gain, "max_distance": max_distance},
        "base_step_support": sum(row["base_supported_steps"] for row in rows) / (adapter.horizon * len(rows)),
        "conditional_calibration": calibration([row["forecasts"]["conditional"] for row in rows if "conditional" in row["forecasts"]]),
        "calibration_common_labeled_supported_scenes": {
            name: calibration([row["forecasts"][name] for row in shared]) for name in score_names
        },
        "paired": {
            name: summarize(rows, name) for name in ("conditional", *alternatives, "finite_oracle")
        },
        "conditional_vs_baselines": {
            name: summarize(rows, "conditional", reference=name) for name in alternatives
        },
        "scenes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="explicit train/evaluation caches, groups and optional plans")
    source.add_argument("--cache-root", type=Path, help="split caches by log and use kinematic plans")
    parser.add_argument("--scenes-per-log", type=int, default=2)
    parser.add_argument("--max-logs", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-evidence", type=Path, help="save bank and held-out base queries for offline diagnostics")
    parser.add_argument("--ego-only", action="store_true", help="omit object annotations and route maps from prediction features")
    args = parser.parse_args()
    if args.scenes_per_log < 1 or args.max_logs < 3 or args.neighbors < 2:
        parser.error("need >=1 scene per log, >=3 logs and >=2 scene neighbors")
    try:
        bank, evaluation = (
            read_manifest(args.manifest) if args.manifest else
            choose_cache_scenes(args.cache_root.resolve(), args.scenes_per_log, args.max_logs, args.seed)
        )
        from ct_sgf_navsim_adapter import EgoOnlyNavsimAdapter, NavsimAdapter

        adapter = EgoOnlyNavsimAdapter() if args.ego_only else NavsimAdapter()
        result = run_pilot(
            adapter, bank, evaluation, neighbors=args.neighbors,
            evidence_path=args.export_evidence,
        )
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
