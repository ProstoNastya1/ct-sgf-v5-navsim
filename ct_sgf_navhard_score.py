"""Score matched NAVSIM navhard_two_stage submissions and retain raw collision times."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import random
import re
import subprocess
from unittest.mock import patch


NAVSIM_COMMIT = "0a380a9063d7162ec93d0f51e9990ebac585f720"
STAGE_KEYS = ("first_stage_predictions", "second_stage_predictions")
AGGREGATE_KEYS = ("stage_one", "stage_two", "combined")
_score_worker = None


def _navsim_path() -> Path:
    import navsim

    path = Path(next(iter(navsim.__path__))).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()
    if commit != NAVSIM_COMMIT:
        raise ValueError(f"expected NAVSIM {NAVSIM_COMMIT}, got {commit}")
    return path


def official_tokens() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the original/synthetic token sets scored by official reactive EPDMS."""
    from omegaconf import OmegaConf

    config = _navsim_path() / "planning/script/config/common/train_test_split/navhard_two_stage.yaml"
    mappings = OmegaConf.load(config).reactive_all_mapping
    stage_one = {token for original, previous, _ in mappings for token in (original, previous)}
    stage_two = {token for _, _, pairs in mappings for pair in pairs for token in pair}
    if not stage_one or not stage_two or stage_one & stage_two:
        raise ValueError("invalid official navhard_two_stage mapping")
    return tuple(sorted(stage_one)), tuple(sorted(stage_two))


def read_submission(path: Path) -> tuple[dict, dict]:
    """Open a trusted local single-seed NAVSIM submission."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    result = []
    for key in STAGE_KEYS:
        submissions = payload[key]
        if not isinstance(submissions, (list, tuple)) or len(submissions) != 1:
            raise ValueError(f"{key} must contain exactly one seed")
        if not isinstance(submissions[0], dict):
            raise ValueError(f"{key} must contain token-to-trajectory mappings")
        result.append(submissions[0])
    return result[0], result[1]


def write_submission(path: Path, first_stage: Mapping, second_stage: Mapping) -> None:
    """Write the prediction-only format accepted by the official scorer."""
    with Path(path).open("wb") as stream:
        pickle.dump({
            STAGE_KEYS[0]: [dict(first_stage)],
            STAGE_KEYS[1]: [dict(second_stage)],
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)


def validate_submission(path: Path, tokens: tuple[tuple[str, ...], tuple[str, ...]]) -> None:
    import numpy as np
    from navsim.common.dataclasses import Trajectory

    for stage, (plans, expected) in enumerate(zip(read_submission(path), tokens), start=1):
        missing = set(expected) - set(plans)
        extra = set(plans) - set(expected)
        if missing or extra:
            raise ValueError(
                f"{path}: stage {stage} token mismatch ({len(missing)} missing, {len(extra)} extra)"
            )
        for token, plan in plans.items():
            if (
                not isinstance(plan, Trajectory)
                or plan.poses.shape != (8, 3)
                or not np.isfinite(plan.poses).all()
                or not math.isclose(plan.trajectory_sampling.interval_length, 0.5)
            ):
                raise ValueError(f"{path}: invalid four-second 2 Hz trajectory at {token}")


def _timestamp_or_none(seconds: float, limit: float) -> float | None:
    if seconds == math.inf:
        return None
    if not math.isfinite(seconds) or seconds < 0.0 or seconds > limit + 1e-6:
        raise ValueError(f"invalid first collision time: {seconds}")
    return seconds


def _trajectory_identity(token, trajectory) -> tuple:
    poses = trajectory.poses
    sampling = trajectory.trajectory_sampling
    return token, sampling.num_poses, sampling.interval_length, poses.dtype.str, poses.tobytes()


def _score_one(official_score, metric_cache, model_trajectory, simulator, scorer, traffic_agents_policy):
    """Record the predicted proposal's raw time before reusing the scorer for the human trajectory."""
    future_sampling = simulator.proposal_sampling
    score_proposals = scorer.score_proposals
    first = []

    def capture_proposals(*args, **kwargs):
        scores = score_proposals(*args, **kwargs)
        if not first:
            if len(scores) != 2:
                raise ValueError("NAVSIM scorer changed its initial proposal layout")
            timestamp = _timestamp_or_none(float(scorer.time_to_at_fault_collision(1)), future_sampling.time_horizon)
            penalty = float(scores[1]["no_at_fault_collisions"].iloc[0])
            if (timestamp is None) != math.isclose(penalty, 1.0):
                raise ValueError("raw collision time and NAVSIM collision penalty disagree")
            first.append(timestamp)
        return scores

    with patch.object(scorer, "score_proposals", capture_proposals):
        result = official_score(
            metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy,
        )
    if len(first) != 1:
        raise ValueError("expected one raw collision record")
    return first[0], result


def _capture_raw_collisions(official, raw: dict, memo: dict | None = None, memo_only: bool = False):
    """Intercept the first result per token before the human filter reuses the scorer."""
    official_run = official.run_pdm_score
    official_score = official.pdm_score

    def run(cfg, first_stage_agent_output, second_stage_agent_output, simulator, scorer, metric_cache_path):
        captures = []

        def capture(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
            if future_sampling != simulator.proposal_sampling:
                raise ValueError("official proposal sampling changed")
            if memo is not None:
                key = _trajectory_identity(str(metric_cache.file_path), model_trajectory)
                if key in memo:
                    timestamp, result_bytes = memo[key]
                    captures.append((metric_cache.log_name, timestamp))
                    return pickle.loads(result_bytes)
            if memo_only:
                raise ValueError("missing trajectory in the parallel score memo")
            timestamp, result = _score_one(
                official_score, metric_cache, model_trajectory, simulator, scorer, traffic_agents_policy,
            )
            captures.append((metric_cache.log_name, timestamp))
            if memo is not None:
                # The upstream CSV/aggregation path mutates rows and nested metric arrays.
                memo[key] = (timestamp, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            return result

        with patch.object(official, "pdm_score", capture):
            score_rows = official_run(
                cfg, first_stage_agent_output, second_stage_agent_output, simulator, scorer, metric_cache_path,
            )
        ordered_tokens = (*first_stage_agent_output, *second_stage_agent_output)
        if len(captures) != len(ordered_tokens) or len(score_rows) != len(ordered_tokens):
            raise ValueError("official scoring missed at least one token; inspect its traceback")
        for token, row, (log, timestamp) in zip(ordered_tokens, score_rows, captures):
            if token != row["token"].iloc[0] or not bool(row["valid"].iloc[0]):
                raise ValueError(f"invalid or reordered official result for {token}")
            raw[token] = (log, timestamp)
        return score_rows

    return run


def _official_config(official):
    from hydra import compose, initialize_config_dir

    config_dir = Path(official.__file__).resolve().parent / official.CONFIG_PATH
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=official.CONFIG_NAME, overrides=["train_test_split=navhard_two_stage"])


def _init_score_worker(cache_path: Path) -> None:
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.script import run_pdm_score_from_submission as official

    global _score_worker
    cfg = _official_config(official)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    # Each arm uses the same configuration. IDM clones its internal state for each replay.
    reactive = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    _score_worker = (MetricCacheLoader(cache_path), official.pdm_score, simulator, scorer, reactive)


def _warm_trajectory(task):
    token, trajectory = task
    loader, official_score, simulator, scorer, reactive = _score_worker
    try:
        metric_cache = loader.get_from_token(token)
        timestamp, result = _score_one(official_score, metric_cache, trajectory, simulator, scorer, reactive)
    except Exception as error:
        raise RuntimeError(f"could not precompute official score for token {token}: {error}") from error
    key = _trajectory_identity(str(metric_cache.file_path), trajectory)
    return key, timestamp, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def warm_score_memo(cache_path: Path, tasks: list, workers: int) -> dict:
    """Calculate the unique official replay rows with isolated process-local simulators."""
    memo = {}
    print(f"Scoring {len(tasks)} distinct trajectories across {workers} CPU workers", flush=True)
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_score_worker, initargs=(cache_path,),
    )
    try:
        for count, (key, timestamp, result) in enumerate(pool.map(_warm_trajectory, tasks, chunksize=4), start=1):
            if key in memo:
                raise ValueError("parallel score keys resolved to the same metric cache and trajectory")
            memo[key] = (timestamp, result)
            if count % 100 == 0 or count == len(tasks):
                print(f"Official replay complete: {count}/{len(tasks)} trajectories", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return memo


def _official_metrics(path: Path, expected: set[str]) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_token = {row["token"]: row for row in rows}
    names = {f"extended_pdm_score_{key}" for key in AGGREGATE_KEYS}
    if len(by_token) != len(rows) or set(by_token) != expected | names:
        raise ValueError("official CSV token coverage differs from the submitted tokens")
    if any(row["valid"].lower() != "true" for row in rows):
        raise ValueError("official CSV contains failed scenarios or invalid aggregation")

    epdms = {}
    for key in AGGREGATE_KEYS:
        value = float(by_token[f"extended_pdm_score_{key}"]["score"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.00000001:
            raise ValueError(f"invalid official EPDMS {key}: {value}")
        epdms[key] = 100 * value
    return epdms


def score_arm(
    path: Path, cache: Path, output: Path,
    tokens: tuple[tuple[str, ...], tuple[str, ...]], memo: dict | None = None, *, memo_only: bool = False,
    mappings=None,
) -> tuple[dict, dict]:
    from navsim.planning.script import run_pdm_score_from_submission as official

    output.mkdir(parents=True, exist_ok=False)
    cfg = _official_config(official)
    if mappings is not None:
        cfg.train_test_split.reactive_all_mapping = mappings
    cfg.output_dir = str(output)
    cfg.experiment_name = output.name
    cfg.metric_cache_path = str(cache)
    cfg.submission_file_path = str(path)
    raw = {}
    with patch.object(official, "run_pdm_score", _capture_raw_collisions(official, raw, memo, memo_only)):
        official.main.__wrapped__(cfg)

    csvs = list(output.glob("*.csv"))
    if len(csvs) != 1:
        raise ValueError(f"expected one official score CSV in {output}")
    epdms = _official_metrics(csvs[0], set(tokens[0]) | set(tokens[1]))
    with (output / "raw_first_at_fault.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("token", "stage", "log_name", "first_at_fault_s"))
        for stage, stage_tokens in enumerate(tokens, start=1):
            writer.writerows((token, stage, *raw[token]) for token in stage_tokens)
    return epdms, raw


def _statistics(raw: dict, tokens: tuple[str, ...]) -> dict:
    event_count = sum(raw[token][1] is not None for token in tokens)
    return {
        "scenarios": len(tokens),
        "raw_at_fault_count": event_count,
        "initial_at_fault_count": sum(raw[token][1] == 0.0 for token in tokens),
        "raw_at_fault_percent": 100 * event_count / len(tokens),
    }


def _paired(before: dict, after: dict, tokens: tuple[str, ...]) -> dict:
    logs: dict[str, list[int]] = {}
    resolved = 0
    introduced = 0
    original_successes = 0
    for token in tokens:
        before_log, before_time = before[token]
        after_log, after_time = after[token]
        if before_log != after_log:
            raise ValueError(f"log name mismatch at {token}")
        previous = before_time is not None
        current = after_time is not None
        resolved += previous and not current
        introduced += not previous and current
        original_successes += not previous
        values = logs.setdefault(before_log, [0, 0])
        values[0] += int(current) - int(previous)
        values[1] += 1

    groups = list(logs.values())
    interval = None
    if len(groups) >= 2:
        rng = random.Random(42)
        samples = []
        for _ in range(3000):
            cluster = rng.choices(groups, k=len(groups))
            samples.append(100 * sum(item[0] for item in cluster) / sum(item[1] for item in cluster))
        samples.sort()
        interval = [samples[75], samples[2924]]
    return {
        "at_fault_change_percentage_points": 100 * (introduced - resolved) / len(tokens),
        "at_fault_change_ci95_clustered_by_log": interval,
        "log_clusters": len(groups),
        "resolved": resolved,
        "introduced": introduced,
        "introduced_percent_of_original_successes": (
            100 * introduced / original_successes if original_successes else None
        ),
    }


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-tokens", action="store_true", help="print the exact official token sets")
    parser.add_argument("--validate-only", action="store_true", help="check all submissions and cache coverage without scoring")
    parser.add_argument("--cache", type=Path, help="official metric cache root with metadata CSV")
    parser.add_argument("--output", type=Path, help="directory for official CSVs and compare.json")
    parser.add_argument("--workers", type=int, default=1, help="parallel official simulator processes")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=SUBMISSION_PKL")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    tokens = official_tokens()
    if arguments.show_tokens:
        print(json.dumps(dict(zip(STAGE_KEYS, tokens))))
        return
    if arguments.cache is None or not arguments.arm or (arguments.output is None and not arguments.validate_only):
        parser.error("provide --cache and one or more --arm; scoring also requires --output")
    arms = {}
    for item in arguments.arm:
        name, separator, value = item.partition("=")
        if not separator or name in arms or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,49}", name):
            parser.error(f"invalid or duplicated arm: {item}")
        arms[name] = Path(value).resolve(strict=True)
        validate_submission(arms[name], tokens)

    from navsim.common.dataloader import MetricCacheLoader

    cache = arguments.cache.resolve(strict=True)
    missing_cache = (set(tokens[0]) | set(tokens[1])) - set(MetricCacheLoader(cache).tokens)
    if missing_cache:
        parser.error(f"official metric cache is missing {len(missing_cache)} tokens")
    if arguments.validate_only:
        print(f"Validated {len(arms)} arms on {len(tokens[0])} first-stage and {len(tokens[1])} second-stage tokens")
        return
    output = arguments.output.resolve()
    reports = {}
    outcomes = {}
    predictions_by_arm = {}
    unique = {}
    for name, submission in arms.items():
        predictions = {}
        for stage in read_submission(submission):
            for token, trajectory in stage.items():
                predictions[_trajectory_identity(token, trajectory)] = (token, trajectory)
        predictions_by_arm[name] = set(predictions)
        unique.update(predictions)
    parallel = arguments.workers > 1
    memo = warm_score_memo(cache, list(unique.values()), arguments.workers) if parallel else {}
    seen = set()
    sections = (("stage_one", tokens[0]), ("stage_two", tokens[1]), ("all_tokens", tokens[0] + tokens[1]))
    for name, submission in arms.items():
        epdms, raw = score_arm(submission, cache, output / name, tokens, memo, memo_only=parallel)
        reports[name] = {
            "submission_sha256": _sha256(submission),
            "epdms": epdms,
            "identical_predictions_reused": len(predictions_by_arm[name] & seen),
            "raw_first_at_fault": {key: _statistics(raw, stage) for key, stage in sections},
        }
        outcomes[name] = raw
        seen.update(predictions_by_arm[name])
        print(f"{name}: official EPDMS {epdms['combined']:.4f}; {len(raw)} valid tokens", flush=True)
    names = list(arms)
    pairs = {}
    for left_index, before in enumerate(names):
        for after in names[left_index + 1:]:
            pairs[f"{after}_minus_{before}"] = {
                "epdms_combined_change": reports[after]["epdms"]["combined"] - reports[before]["epdms"]["combined"],
                "raw_first_at_fault": {
                    key: _paired(outcomes[before], outcomes[after], stage) for key, stage in sections
                },
            }
    (output / "compare.json").write_text(json.dumps({
        "navsim_commit": NAVSIM_COMMIT,
        "split": "navhard_two_stage",
        "arms": reports,
        "paired": pairs,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
