"""Score a small, explicitly restricted NAVSIM two-stage diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ct_sgf import Rollout
from ct_sgf_baselines import TrajectoryLogistic
from ct_sgf_navhard_experiment import EvidenceScene, _init_policy_worker, _policy_prediction, disjoint_bank
from ct_sgf_navhard_score import (
    _navsim_path, _paired, _statistics, _trajectory_identity, read_submission, score_arm, warm_score_memo,
    write_submission,
)
from ct_sgf_navsim_adapter import EgoOnlyNavsimAdapter


def cached_pairs(cache, limit):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_navsim_path() / "planning/script/config/common/train_test_split/navhard_two_stage.yaml")
    now = time.time()
    available = {
        path.parent.name: path.resolve()
        for path in cache.glob("*/*/*/metric_cache.pkl")
        if path.stat().st_size > 100 and path.stat().st_mtime < now - 15
    }
    groups = []
    by_stage = [set(), set()]
    for left, right, all_pairs in sorted(cfg.reactive_all_mapping, key=lambda group: tuple(group[:2])):
        if not {left, right} <= available.keys() or {left, right} & by_stage[0]:
            continue
        pairs = []
        within = set()
        for pair in sorted(tuple(pair) for pair in all_pairs):
            if len(pair) == 2 and set(pair) <= available.keys() and not set(pair) & (by_stage[1] | within):
                pairs.append(list(pair))
                within.update(pair)
            if len(pairs) == 2:
                break
        if len(pairs) < 2:
            continue
        groups.append([left, right, pairs])
        by_stage[0].update((left, right))
        by_stage[1].update(within)
        if len(groups) == limit:
            break
    if len(groups) != limit:
        raise ValueError("not enough cached original pairs with two complete synthetic pairs")
    return groups, tuple(tuple(sorted(tokens)) for tokens in by_stage), available


def frozen_bank(spool, loader, stages):
    if (spool / "navtrain_bank.json").exists():
        entries = json.loads((spool / "navtrain_bank.json").read_text(encoding="utf-8"))["bank"]
    else:
        entries = []
        for path in sorted(spool.glob("navtrain_labels_*/*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if "rollouts" in row:
                entries.append(row)
    locations = {}
    for token in (*stages[0], *stages[1]):
        cache = loader.get_from_token(token)
        pose = cache.ego_state.rear_axle
        locations.setdefault(cache.map_parameters.map_name, []).append((pose.x, pose.y))
    scenes = []
    for row in entries:
        context = tuple(row["context"])
        rollouts = tuple(
            Rollout(row["group"], [context + tuple(step) for step in record["features"]],
                    record["observed_steps"], record["first_event"])
            for record in row["rollouts"]
        )
        scenes.append(EvidenceScene(row["group"], row["map_name"], (row["x"], row["y"]), context, rollouts))
    return disjoint_bank(scenes, locations)


def run(args):
    from navsim.common.dataloader import MetricCacheLoader

    groups, stages, available = cached_pairs(args.cache, args.groups)
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "local_cache"
    metadata = cache / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "pilot.csv").write_text(
        "file_name\n" + "\n".join(str(available[token]) for token in (*stages[0], *stages[1])) + "\n",
        encoding="utf-8",
    )
    (args.output / "fixed_subset.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    loader = MetricCacheLoader(cache)
    bank = frozen_bank(args.inputs, loader, stages)
    adapter = EgoOnlyNavsimAdapter()
    widths = adapter.context_bandwidths + adapter.bandwidths
    scorer = TrajectoryLogistic.fit([record for scene in bank for record in scene.rollouts], widths, 40)
    if scorer is None:
        raise ValueError("the available location-disjoint training data needs both outcomes")
    frozen = read_submission(args.inputs / "ego_status_mlp.pkl")
    arms = {"EgoStatusMLP": [{}, {}], "CT-SGF": [{}, {}], "LearnedScorer": [{}, {}]}
    _init_policy_worker(cache, bank, scorer, 12, 3.0, 0.02)
    added = {}
    for index, tokens in enumerate(stages):
        for token in tokens:
            original = frozen[index][token]
            _, _, conditional, learned, _ = _policy_prediction((index, token, original))
            for name, plan in zip(arms, (original, conditional, learned)):
                arms[name][index][token] = plan
                added[_trajectory_identity(token, plan)] = (token, plan)
    memo = warm_score_memo(cache, list(added.values()), args.workers)
    summary = {}
    raw = {}
    for name, predictions in arms.items():
        path = args.output / f"{name}.pkl"
        write_submission(path, *predictions)
        epdms, raw[name] = score_arm(path, cache, args.output / name, stages, memo, memo_only=True, mappings=groups)
        summary[name] = {"epdms_for_subset": epdms, "original": _statistics(raw[name], stages[0]),
                         "synthetic": _statistics(raw[name], stages[1]),
                         "all": _statistics(raw[name], (*stages[0], *stages[1]))}
    result = {
        "scope": "Cache-ready original pairs and two cached synthetic pairs per mapping. NAVSIM two-stage aggregation restricted to those pairs.",
        "train_scenes": len(bank), "original_scenes": len(stages[0]), "synthetic_scenes": len(stages[1]),
        "arms": summary, "ct_vs_scorer_original": _paired(raw["LearnedScorer"], raw["CT-SGF"], stages[0]),
    }
    (args.output / "compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
