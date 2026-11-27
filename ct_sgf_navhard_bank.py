"""Build the fixed train-only evidence bank for the NAVSIM navhard study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import subprocess
import sys

from ct_sgf import Rollout
from ct_sgf_navsim import parse_plan, within_pilot_bounds
from ct_sgf_navsim_adapter import EgoOnlyNavsimAdapter, InitialCollisionError, NAVSIM_COMMIT


CANDIDATE_FACTORS = (1.0, 0.65, 0.8, 0.9, 1.1, 1.2, 1.35)
NAVTRAIN_FILTER = "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
LOG_PATTERN = re.compile(r"(\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+)_\d+_\d+\Z")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+\Z")


def candidate_plans(base) -> tuple:
    """Keep the frozen plan first, followed by unique longitudinal scale variants."""
    original = parse_plan(base)
    return tuple(dict.fromkeys(
        tuple((factor * x, factor * y, yaw) for x, y, yaw in original)
        for factor in CANDIDATE_FACTORS
    ))


def load_planner(checkpoint: Path):
    from navsim.agents.ego_status_mlp_agent import EgoStatusMLPAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    model = EgoStatusMLPAgent(
        hidden_layer_dim=512, lr=1e-4, checkpoint_path=str(checkpoint),
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    )
    model.initialize()
    model.eval()
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recording_group(log_name: str) -> str:
    match = LOG_PATTERN.fullmatch(log_name)
    if match is None:
        raise ValueError(f"unrecognized NAVSIM log name: {log_name}")
    return match.group(1)


def select_source(data_root: Path, scene_filter, size: int, development_size: int, seed: int):
    """Choose non-overlapping windows deterministically through the official filter."""
    from navsim.common.dataloader import filter_scenes

    log_dir = data_root / "navsim_logs" / "trainval"
    expected_logs = set(scene_filter.log_names)
    if not expected_logs or not scene_filter.tokens:
        raise ValueError("the pinned navtrain filter must specify official logs and tokens")
    available = {file.stem for file in log_dir.glob("*.pkl") if file.is_file()}
    missing = expected_logs - available
    if missing:
        raise ValueError(f"official trainval extraction is incomplete: {len(missing)} navtrain logs missing")
    logs = sorted(expected_logs)
    rng = random.Random(seed)
    rng.shuffle(logs)
    splits = {"bank": [], "development": []}
    used_groups = set()
    frame_gap = scene_filter.num_history_frames + scene_filter.num_future_frames
    for log_name in logs:
        group = recording_group(log_name)
        if group in used_groups:
            continue
        split = "bank" if len(splits["bank"]) < size else "development"
        target = size if split == "bank" else development_size
        if len(splits[split]) == target:
            break
        # Keep the official stride and token allow-list. Thinning comes after
        # matching, so valid windows retain the official alignment.
        filtered, _ = filter_scenes(
            log_dir, replace(scene_filter, log_names=[log_name], max_scenes=None),
        )
        if not filtered:
            continue
        used_groups.add(group)
        limit = min(2, target - len(splits[split]))
        selected_times = []
        entries = list(filtered.items())
        rng.shuffle(entries)
        for token, frames in entries:
            if not TOKEN_PATTERN.fullmatch(token):
                raise ValueError(f"unrecognized NAVSIM token: {token}")
            current = frames[scene_filter.num_history_frames - 1]
            if current["log_name"] != log_name:
                raise ValueError(f"scene {token} disagrees with its source log")
            timestamp = int(current["timestamp"])
            # OpenScene is sampled at 2 Hz. Leave an entire raw frame window
            # between starts when selecting twice from a recording.
            min_separation_us = frame_gap * 500_000
            if any(abs(timestamp - prev) < min_separation_us for prev in selected_times):
                continue
            selected_times.append(timestamp)
            splits[split].append({"token": token, "log": log_name, "group": group})
            if len(selected_times) == limit:
                break
    if len(splits["bank"]) != size or len(splits["development"]) != development_size:
        raise ValueError("not enough eligible navtrain source scenes across disjoint recordings")
    print(f"Fixed navtrain selection: {size} bank and {development_size} development scenes", flush=True)
    return splits


def existing_caches(cache_root: Path, selected: list[dict]) -> tuple[dict, list[dict]]:
    caches = {}
    missing = []
    for row in selected:
        matches = sorted(path for path in (cache_root / row["log"]).glob(
            f"*/{row['token']}/metric_cache.pkl",
        ) if path.is_file())
        if len(matches) > 1:
            raise ValueError(f"duplicate cache entries for {row['token']}")
        if matches:
            caches[row["token"]] = matches[0]
        else:
            missing.append(row)
    return caches, missing


def ensure_caches(cache_root: Path, selected: list[dict], workers: int) -> dict:
    caches, missing = existing_caches(cache_root, selected)
    if not missing:
        return caches
    logs = sorted({row["log"] for row in missing})
    tokens = sorted(row["token"] for row in missing)
    subprocess.run([
        sys.executable, "-m", "navsim.planning.script.run_metric_caching",
        "train_test_split=navtrain", "gpu=false", "worker=single_machine_thread_pool",
        f"worker.max_workers={workers}", "force_feature_computation=false",
        f"metric_cache_path={json.dumps(str(cache_root), ensure_ascii=False)}",
        f"train_test_split.scene_filter.log_names=[{','.join(logs)}]",
        f"train_test_split.scene_filter.tokens=[{','.join(tokens)}]",
        f"train_test_split.scene_filter.max_scenes={len(tokens)}",
    ], check=True)
    caches, missing = existing_caches(cache_root, selected)
    if missing:
        raise ValueError(f"official caching omitted {len(missing)} selected navtrain scenes")
    return caches


def label_scene(adapter, row, path, candidates):
    token = row["token"]
    cache = adapter.load_cache(path)
    if cache.log_name != row["log"]:
        raise ValueError(f"cache provenance disagrees with the navtrain selection for {token}")
    rollouts = []
    plans = []
    feasible = []
    try:
        for candidate_index, plan in enumerate(candidates):
            in_bounds = within_pilot_bounds(adapter.tracked_candidate(cache, plan))
            if candidate_index != 0 and not in_bounds:
                continue
            outcome = adapter.assess(cache, plan)
            rollouts.append(asdict(Rollout(
                row["group"], outcome.features, outcome.observed_steps, outcome.first_event,
            )))
            plans.append(plan)
            feasible.append(in_bounds)
    except InitialCollisionError:
        return {**row, "reason": "at_fault_collision_before_forecast"}
    rear_axle = cache.ego_state.rear_axle
    x, y = float(rear_axle.x), float(rear_axle.y)
    context = adapter.scene_context(cache)
    if not all(map(math.isfinite, (*context, x, y))):
        raise ValueError(f"non-finite ego context in {token}")
    return {
        **row, "map_name": cache.map_parameters.map_name, "x": x, "y": y, "context": context,
        "rollouts": rollouts, "plans": plans, "tracked_bounds_met": feasible,
    }


_label_adapter = None


def _init_label_worker():
    global _label_adapter
    _label_adapter = EgoOnlyNavsimAdapter(traffic="reactive")


def _label_worker(task):
    return label_scene(_label_adapter, *task)


def collect_split(adapter, model, loader, caches, selected, name, workers, progress_dir):
    results = {}
    tasks = []
    for row in selected:
        token = row["token"]
        saved = progress_dir / f"{token}.json"
        if saved.is_file():
            results[token] = json.loads(saved.read_text(encoding="utf-8"))
        else:
            trajectory = model.compute_trajectory(loader.get_agent_input_from_token(token))
            tasks.append((row, caches[token], candidate_plans(trajectory.poses)))

    def record(result):
        token = result["token"]
        saved = progress_dir / f"{token}.json"
        temporary = saved.with_suffix(".part")
        temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        temporary.replace(saved)
        results[token] = result
        print(f"navtrain {name} {len(results)}/{len(selected)}: {len(result.get('rollouts', []))} candidates", flush=True)

    if tasks and workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_label_worker,
        ) as pool:
            for future in as_completed([pool.submit(_label_worker, task) for task in tasks]):
                record(future.result())
    else:
        for task in tasks:
            record(label_scene(adapter, *task))
    ordered = [results[row["token"]] for row in selected]
    return [row for row in ordered if "rollouts" in row], [row for row in ordered if "reason" in row]


def run_bank(args):
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from omegaconf import OmegaConf

    devkit = Path(os.environ["NAVSIM_DEVKIT_ROOT"]).resolve()
    observed_commit = subprocess.check_output(["git", "-C", str(devkit), "rev-parse", "HEAD"], text=True).strip()
    if observed_commit != NAVSIM_COMMIT:
        raise ValueError(f"expected NAVSIM commit {NAVSIM_COMMIT}, found {observed_commit}")
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"]).resolve()
    scene_filter = instantiate(OmegaConf.load(devkit / NAVTRAIN_FILTER))
    splits = select_source(data_root, scene_filter, args.scenes, args.development_scenes, args.seed)
    selected = [*splits["bank"], *splits["development"]]
    caches = ensure_caches(args.cache.resolve(), selected, args.workers)
    source_loader = SceneLoader(
        data_path=data_root / "navsim_logs" / "trainval", original_sensor_path=None,
        sensor_config=SensorConfig.build_no_sensors(),
        scene_filter=replace(
            scene_filter, max_scenes=len(selected),
            log_names=sorted({row["log"] for row in selected}),
            tokens=sorted(row["token"] for row in selected),
        ),
    )
    if set(source_loader.tokens) != {row["token"] for row in selected}:
        raise ValueError("the input loader did not recreate the fixed source token set")
    model = load_planner(args.checkpoint.resolve())
    adapter = EgoOnlyNavsimAdapter(traffic="reactive")
    checkpoint_digest = sha256(args.checkpoint)
    identity = f"{NAVSIM_COMMIT}:{checkpoint_digest}:seed{args.seed}:reactive:{CANDIDATE_FACTORS}"
    progress_dir = args.output.parent / f"navtrain_labels_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    progress_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "split": "navtrain", "navsim_commit": NAVSIM_COMMIT,
        "checkpoint_sha256": checkpoint_digest, "candidate_factors": CANDIDATE_FACTORS,
        "seed": args.seed, "horizon": adapter.horizon,
        "traffic_policy": "official_reactive_idm",
        "context_bandwidths": adapter.context_bandwidths, "feature_bandwidths": adapter.bandwidths,
        "selection": splits, "skipped": [],
    }
    for split, rows in splits.items():
        output[split], skipped = collect_split(
            adapter, model, source_loader, caches, rows, split, args.label_workers, progress_dir,
        )
        output["skipped"].extend({"split": split, **row} for row in skipped)
    if not output["bank"]:
        raise ValueError("all selected bank scenes were invalid")
    rollouts = [rollout for row in output["bank"] for rollout in row["rollouts"]]
    if len({rollout["first_event"] is not None for rollout in rollouts}) != 2:
        raise ValueError("the fixed training bank has only one collision class; a trained comparison is undefined")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(f"Wrote {len(rollouts)} labelled bank rollouts to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", type=int, default=64)
    parser.add_argument("--development-scenes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label-workers", type=int, default=1)
    args = parser.parse_args()
    if args.scenes < 2 or args.development_scenes < 0 or min(args.workers, args.label_workers) < 1:
        parser.error("need two or more evidence scenes and at least one worker")
    try:
        run_bank(args)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
