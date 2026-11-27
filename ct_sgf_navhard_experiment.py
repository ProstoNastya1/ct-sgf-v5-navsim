"""Build matched NAVSIM navhard submissions without accessing held-out outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

from ct_sgf import Rollout
from ct_sgf_baselines import Risk, StepwiseLogistic, TrajectoryLogistic, TrajectoryLogKDE
from ct_sgf_evaluate import calibration
from ct_sgf_navsim import parse_plan, within_pilot_bounds
from ct_sgf_navsim_adapter import NAVSIM_COMMIT, EgoOnlyNavsimAdapter


@dataclass
class EvidenceScene:
    group: str
    map_name: str
    position: tuple[float, float]
    context: tuple[float, ...]
    rollouts: tuple[Rollout, ...]


def read_bank(path, adapter):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["split"] != "navtrain" or payload["horizon"] != adapter.horizon:
        raise ValueError("evidence must come from the official navtrain split at the same horizon")
    if payload["navsim_commit"] != NAVSIM_COMMIT:
        raise ValueError("navtrain labels must be computed with the pinned NAVSIM revision")
    if payload["traffic_policy"] != "official_reactive_idm":
        raise ValueError("the navtrain bank must use the same reactive traffic policy as navhard")
    if tuple(payload["context_bandwidths"]) != tuple(adapter.context_bandwidths):
        raise ValueError("inconsistent context bandwidths in navtrain bank")
    if tuple(payload["feature_bandwidths"]) != tuple(adapter.bandwidths):
        raise ValueError("inconsistent transition bandwidths in navtrain bank")
    records = []
    for row in payload["bank"]:
        context = tuple(row["context"])
        rollouts = tuple(
            Rollout(
                row["group"], [context + tuple(features) for features in record["features"]],
                record["observed_steps"], record["first_event"],
            )
            for record in row["rollouts"]
        )
        records.append(EvidenceScene(
            row["group"], row["map_name"], (float(row["x"]), float(row["y"])), context, rollouts,
        ))
    return records, payload


def disjoint_bank(bank, locations, *, minimum_distance=150.0):
    """Use only train scenes outside the radius of every held-out scene on its map."""
    locations = {name: np.asarray(points, dtype=float) for name, points in locations.items()}
    kept = []
    for scene in bank:
        target = locations.get(scene.map_name)
        if target is None or np.all(np.linalg.norm(target - scene.position, axis=1) >= minimum_distance):
            kept.append(scene)
    return kept


class SceneEvidence:
    """Local Eq. (3) kernel on transition features, with independent-log support."""

    def __init__(self, bank, widths, progress_idx, *, success_only=False):
        rows = [(features, event, record.group) for scene in bank for record in scene.rollouts
                if not success_only or (record.first_event is None and record.observed_steps == len(record.features))
                for features, event in record.at_risk()]
        self.widths = np.asarray(widths)
        self.features = np.asarray([features for features, _, _ in rows], dtype=float).reshape(-1, len(widths))
        self.normalized = self.features / self.widths
        self.bank_norms = np.einsum("ij,ij->i", self.normalized, self.normalized)
        self.event = np.asarray([event for _, event, _ in rows], dtype=float)
        self.groups = [np.asarray([row[2] == group for row in rows])
                       for group in sorted({row[2] for row in rows})]
        self.progress_idx = progress_idx

    def predict(self, features):
        queries = np.asarray(features, dtype=float)
        normalized = queries / self.widths
        norms = np.einsum("ij,ij->i", normalized, normalized)
        squared = norms[:, None] + self.bank_norms - 2 * (normalized @ self.normalized.T)
        weights = np.exp(-0.5 * np.maximum(squared, 0.0))
        progress_delta = queries[:, self.progress_idx, None] - self.features[:, self.progress_idx]
        weights[np.abs(progress_delta) > 12.0] = 0.0
        exposure = weights.sum(axis=1)
        squared_exposure = np.einsum("ij,ij->i", weights, weights)
        ess = np.divide(
            exposure * exposure, squared_exposure, out=np.zeros_like(exposure),
            where=squared_exposure > 0,
        )
        groups = np.zeros(len(queries), dtype=int)
        for mask in self.groups:
            groups += weights[:, mask].sum(axis=1) >= 0.25
        supported = (exposure >= 1.0) & (ess >= 2.0) & (groups >= 2)
        events = weights @ self.event
        hazard = (0.25 + events) / (5.0 + exposure)
        return float(-np.log1p(-hazard).sum()), int(supported.sum())


def nearest_scenes(bank, context, widths, count):
    scaled = tuple(value / width for value, width in zip(context, widths))
    return sorted(bank, key=lambda scene: math.dist(
        tuple(value / width for value, width in zip(scene.context, widths)),
        scaled,
    ))[:count]


def squared_plan_distance(original, plan):
    delta = np.asarray(plan) - np.asarray(original)
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    return float(np.square(delta / np.asarray((4.0, 1.0, 0.3))).sum())


def select_candidate(eligible, column, min_gain):
    if not eligible:
        return 0
    best = min(eligible, key=lambda item: item[column])
    return best[0] if eligible[0][column] - best[column] > min_gain else 0


class BoostedTrajectoryScorer:
    """Nonlinear trajectory comparator trained on the same flattened evidence."""

    def __init__(self, records, widths):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.widths = np.asarray(widths)
        self.training_count = len(records)
        inputs = [self.vector(record.features) for record in records]
        labels = [record.first_event is not None for record in records]
        self.estimator = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=8, l2_regularization=0.1,
            early_stopping=False, random_state=42,
        ).fit(inputs, labels)

    def vector(self, features):
        return (np.asarray(features) / self.widths).ravel()

    def predict(self, features):
        score = float(self.estimator.decision_function([self.vector(features)])[0])
        return Risk.from_logit(score)


def choose_learned_scorer(bank, development, widths, horizon):
    records = [
        record for scene in bank for record in scene.rollouts
        if len(record.features) == horizon and (record.first_event is not None or record.observed_steps == horizon)
    ]
    logistic = TrajectoryLogistic.fit(records, widths, horizon)
    if logistic is None:
        raise ValueError("the disjoint navtrain bank needs both collision and no-collision trajectories")
    variants = {"logistic": logistic, "boosted_trees": BoostedTrajectoryScorer(records, widths)}
    dev_samples = []
    for row in development:
        context = tuple(row["context"])
        for record in row["rollouts"]:
            failed = record["first_event"] is not None
            if len(record["features"]) == horizon and (failed or record["observed_steps"] == horizon):
                dev_samples.append((tuple(context + tuple(features) for features in record["features"]), failed))
    loss = {}
    if dev_samples:
        for name, model in variants.items():
            losses = []
            for features, failed in dev_samples:
                forecast = model.predict(features)
                losses.append(-forecast.log_failure if failed else -forecast.log_survival)
            loss[name] = fmean(losses)
    selected = min(loss, key=loss.get) if loss else "logistic"
    return variants[selected], {
        "selected": selected, "development_trajectory_count": len(dev_samples),
        "development_log_loss": loss,
    }


def correct_token(adapter, cache, original, plans, near, learned, widths, *, eta, min_gain):
    context = tuple(adapter.scene_context(cache))
    progress = len(context) + adapter.feature_names.index(adapter.progress_feature)
    evidence = SceneEvidence(near, widths, progress)
    eligible = []
    base_support = 0
    for index, plan in enumerate(plans):
        distance2 = squared_plan_distance(original, plan)
        if distance2 > 36.0:
            continue
        tracked = adapter.tracked_candidate(cache, plan)
        features = tuple(context + tuple(row) for row in tracked.features)
        conditional_risk, support = evidence.predict(features)
        if index == 0:
            base_support = support
        if not within_pilot_bounds(tracked) or support != len(features):
            if index == 0:
                return original, original, base_support, 0
            continue
        scorer_risk = -learned.predict(features).log_survival
        prior = 0.5 * distance2
        eligible.append((index, prior + eta * conditional_risk, prior + eta * scorer_risk))
    return plans[select_candidate(eligible, 1, min_gain)], plans[select_candidate(eligible, 2, min_gain)], base_support, len(eligible)


def development_report(rows, bank, learned, adapter, args, evaluation_logs):
    """Estimate paired log variance with labels from disjoint navtrain recordings."""
    widths = adapter.context_bandwidths + adapter.bandwidths
    progress = len(adapter.context_bandwidths) + adapter.feature_names.index(adapter.progress_feature)
    train_groups = {scene.group for scene in bank}
    per_recording = defaultdict(list)
    supported = []
    for row in rows:
        if row["group"] in train_groups:
            raise ValueError("training and development share a recording")
        context = tuple(row["context"])
        near = nearest_scenes(bank, context, adapter.context_bandwidths, args.neighbors)
        evidence = SceneEvidence(near, widths, progress)
        success_evidence = SceneEvidence(near, widths, progress, success_only=True)
        plans = row["plans"]
        available = []
        success_available = []
        feasible_by_distance = []
        shared_base_supported = False
        success_base_supported = False
        for index, (record, plan, feasible) in enumerate(zip(row["rollouts"], plans, row["tracked_bounds_met"])):
            features = tuple(context + tuple(transition) for transition in record["features"])
            negative_log_survival, support = evidence.predict(features)
            success_risk, success_support = success_evidence.predict(features)
            if index == 0:
                supported.append(support)
                shared_base_supported = feasible and support == adapter.horizon
                success_base_supported = feasible and success_support == adapter.horizon
            distance2 = squared_plan_distance(plans[0], plan)
            if not feasible or distance2 > 36.0:
                continue
            prior = 0.5 * distance2
            failed = record["first_event"] is not None
            if failed or record["observed_steps"] == adapter.horizon:
                feasible_by_distance.append((failed, distance2, index))
            if shared_base_supported and support == adapter.horizon:
                available.append((index, prior + args.eta * negative_log_survival,
                                  prior - args.eta * learned.predict(features).log_survival))
            if success_base_supported and success_support == adapter.horizon:
                success_available.append((index, prior + args.eta * success_risk))
        oracle = min(feasible_by_distance)[2] if feasible_by_distance and row["tracked_bounds_met"][0] else 0
        indices = (
            0, select_candidate(available, 1, args.min_gain), select_candidate(available, 2, args.min_gain),
            select_candidate(success_available, 1, args.min_gain), oracle,
        )
        per_recording[row["group"]].append(tuple(
            int(row["rollouts"][index]["first_event"] is not None) for index in indices
        ))
    deltas = [fmean(item[1] - item[2] for item in pairs) for pairs in per_recording.values()]
    outcomes = [item for group in per_recording.values() for item in group]
    standard_deviation = stdev(deltas) if len(deltas) > 1 else None
    return {
        "development_scenes": len(outcomes), "development_recordings": len(per_recording),
        "original_stage_evaluation_logs": evaluation_logs,
        "base_supported_steps": sum(supported),
        "paired_recording_standard_deviation": standard_deviation,
        "approximate_mde_80_power_5_percent": (
            (1.959964 + 0.841621) * standard_deviation / math.sqrt(evaluation_logs)
            if standard_deviation and evaluation_logs else None
        ),
        "collision_counts": {
            name: sum(items[index] for items in outcomes)
            for index, name in enumerate(("EgoStatusMLP", "CT-SGF", "LearnedScorer", "SuccessOnly", "FiniteOracle"))
        },
    }


def development_curves(rows, bank, adapter, neighbors):
    """Use nested recording groups and score all models on common supported labels."""
    widths = adapter.context_bandwidths + adapter.bandwidths
    progress = len(adapter.context_bandwidths) + adapter.feature_names.index(adapter.progress_feature)
    groups = list(dict.fromkeys(scene.group for scene in bank))
    sizes = sorted({len(groups), *(size for size in (4, 8, 16) if size <= len(groups))})
    reports = []
    for size in sizes:
        chosen = set(groups[:size])
        subset = [scene for scene in bank if scene.group in chosen]
        records = [record for scene in subset for record in scene.rollouts]
        variants = {
            model: cls.fit(records, widths, adapter.horizon)
            for model, cls in (("log_kde", TrajectoryLogKDE), ("trajectory", TrajectoryLogistic),
                               ("stepwise", StepwiseLogistic))
        }
        common = {name: [] for name in ("conditional", *variants)}
        supported_steps = total_steps = 0
        for row in rows:
            context = tuple(row["context"])
            near = nearest_scenes(subset, context, adapter.context_bandwidths, neighbors)
            evidence = SceneEvidence(near, widths, progress)
            for record in row["rollouts"]:
                features = tuple(context + tuple(transition) for transition in record["features"])
                negative_log_survival, support = evidence.predict(features)
                supported_steps += support
                total_steps += len(features)
                failed = record["first_event"] is not None
                known = failed or record["observed_steps"] == adapter.horizon
                if not known or support != adapter.horizon or any(model is None for model in variants.values()):
                    continue
                probability = -math.expm1(-negative_log_survival)
                common["conditional"].append((probability, -negative_log_survival, failed))
                for name, model in variants.items():
                    score = model.predict(features)
                    common[name].append((score.failure_probability, score.log_survival, failed, score.log_failure))
        reports.append({
            "recordings": size, "trajectories": len(records), "supported_steps": supported_steps,
            "total_steps": total_steps, "available_scorers": [name for name, model in variants.items() if model],
            "common_subset_calibration": {name: calibration(rows) for name, rows in common.items()},
        })
    return reports


_policy_state = None


def _init_policy_worker(cache_path, bank, learned, neighbors, eta, min_gain):
    from navsim.common.dataloader import MetricCacheLoader

    global _policy_state
    adapter = EgoOnlyNavsimAdapter()
    _policy_state = (MetricCacheLoader(cache_path), adapter, bank, learned, neighbors, eta, min_gain)


def _policy_prediction(task):
    from navsim.common.dataclasses import Trajectory
    from ct_sgf_navhard_bank import candidate_plans

    index, token, base = task
    loader, adapter, bank, learned, neighbors, eta, min_gain = _policy_state
    cache = loader.get_from_token(token)
    context = adapter.scene_context(cache)
    near = nearest_scenes(bank, context, adapter.context_bandwidths, neighbors)
    original = parse_plan(base.poses)
    plans = candidate_plans(original)
    ct, scorer, support, count = correct_token(
        adapter, cache, original, plans, near, learned, adapter.context_bandwidths + adapter.bandwidths,
        eta=eta, min_gain=min_gain,
    )

    def trajectory(selected):
        if selected == original:
            return base
        return Trajectory(poses=np.asarray(selected), trajectory_sampling=base.trajectory_sampling)

    diagnostics = {
        "token": token, "stage": index + 1, "log": cache.log_name, "base_supported_steps": support,
        "shared_supported_candidates": count, "ct_changed": ct != original,
        "scorer_changed": scorer != original,
    }
    return index, token, trajectory(ct), trajectory(scorer), diagnostics


def write_predictions(args):
    from navsim.common.dataloader import MetricCacheLoader

    from ct_sgf_navhard_score import official_tokens, read_submission, write_submission

    adapter = EgoOnlyNavsimAdapter()
    bank, payload = read_bank(args.bank, adapter)
    stages = read_submission(args.base)
    expected = official_tokens()
    if any(set(submitted) != set(tokens) for submitted, tokens in zip(stages, expected)):
        raise ValueError("base submission must include exactly the two official reactive stage token sets")
    loader = MetricCacheLoader(args.cache)
    tokens = tuple(token for stage in expected for token in stage)
    locations = {}
    stage_one = set(expected[0])
    original_logs = set()
    for token in tokens:
        cache = loader.get_from_token(token)
        pose = cache.ego_state.rear_axle
        map_name = cache.map_parameters.map_name
        point = (float(pose.x), float(pose.y))
        locations.setdefault(map_name, []).append(point)
        if token in stage_one:
            original_logs.add(cache.log_name)
    for row in payload["development"]:
        locations.setdefault(row["map_name"], []).append((float(row["x"]), float(row["y"])))
    eligible_bank = disjoint_bank(bank, locations)
    if len({scene.group for scene in eligible_bank}) < 2:
        raise ValueError("fewer than two navtrain log groups remain after the geographic holdout")
    widths = adapter.context_bandwidths + adapter.bandwidths
    learned, selection = choose_learned_scorer(eligible_bank, payload["development"], widths, adapter.horizon)
    args.output.mkdir(parents=True, exist_ok=True)
    dev = development_report(payload["development"], eligible_bank, learned, adapter, args, len(original_logs))
    dev["bank_size_curves"] = development_curves(payload["development"], eligible_bank, adapter, args.neighbors)
    (args.output / "train_only_development.json").write_text(
        json.dumps(dev, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    conditional = [{}, {}]
    comparator = [{}, {}]
    reports = []
    tasks = [(index, token, stage[token]) for index, stage in enumerate(stages) for token in expected[index]]
    initializer = (args.cache, eligible_bank, learned, args.neighbors, args.eta, args.min_gain)

    def collect(results):
        for count, (index, token, ct, scorer, diagnostics) in enumerate(results, start=1):
            conditional[index][token] = ct
            comparator[index][token] = scorer
            reports.append(diagnostics)
            if count % 100 == 0 or count == len(tasks):
                print(f"Policy inference: {count}/{len(tasks)}", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_policy_worker, initargs=initializer,
        ) as pool:
            collect(pool.map(_policy_prediction, tasks, chunksize=4))
    else:
        _init_policy_worker(*initializer)
        collect(map(_policy_prediction, tasks))
    write_submission(args.output / "ct_sgf.pkl", *conditional)
    write_submission(args.output / "learned_scorer.pkl", *comparator)
    summary = {
        "train_bank_scenes": len(bank), "geo_disjoint_bank_scenes": len(eligible_bank),
        "geo_exclusion_radius_m": 150.0, "trajectory_training_count": learned.training_count,
        "eta": args.eta, "min_gain": args.min_gain, "neighbors": args.neighbors,
        "training_checkpoint_sha256": payload["checkpoint_sha256"], "tokens": reports,
        "learned_scorer_selection": selection,
        "train_only_development": dev,
    }
    (args.output / "shared_candidate_diagnostics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def export_base(checkpoint, output):
    import os

    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.planning.script.run_create_submission_pickle import run_test_evaluation

    from ct_sgf_navhard_bank import load_planner
    from ct_sgf_navhard_score import official_tokens, write_submission

    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_create_submission_pickle", overrides=["train_test_split=navhard_two_stage"])
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    stages = run_test_evaluation(
        cfg, load_planner(checkpoint), instantiate(cfg.train_test_split.scene_filter),
        data_root / "navsim_logs/test", data_root / "navhard_two_stage/sensor_blobs",
        data_root / "sensor_blobs/test", data_root / "navhard_two_stage/synthetic_scene_pickles",
    )
    required = official_tokens()
    if any(set(tokens) - stage.keys() for stage, tokens in zip(stages, required)):
        raise ValueError("official MLP export failed to predict every required stage token")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output, *({token: stage[token] for token in tokens} for stage, tokens in zip(stages, required)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-base", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--min-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.export_base:
        export_base(args.export_base, args.output)
        return
    if args.bank is None or args.base is None or args.cache is None:
        parser.error("policy inference requires --bank, --base and --cache")
    if args.workers < 1 or args.neighbors < 2 or any(not math.isfinite(value) or value < 0 for value in (args.eta, args.min_gain)):
        parser.error("need at least two neighbors and finite, nonnegative correction settings")
    write_predictions(args)


if __name__ == "__main__":
    main()
