"""Select a bounded, map-disjoint NAVSIM mini pilot and resolve a cache manifest.

Use only official NAVSIM logs and metric caches from a trusted source. NAVSIM's
official input processors read pickles. The optional caching pass uses only the
selected original-scene tokens and does not load camera or LiDAR files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


NUM_HISTORY = 4
NUM_FUTURE = 10
FRAME_INTERVAL = NUM_HISTORY + NUM_FUTURE
CACHE_FILENAME = "metric_cache.pkl"
CACHE_MODULE = "navsim.planning.script.run_metric_caching"
SAFE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
SPLITS = ("bank", "evaluation")


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"invalid NAVSIM {name} {value!r}")
    return value


def _scene_frames(log_dir: Path, log_name: str, max_scenes: int):
    try:
        from navsim.common.dataloader import SceneFilter, filter_scenes
    except ImportError as error:
        raise RuntimeError("scene selection needs the pinned NAVSIM environment") from error
    scene_filter = SceneFilter(
        log_names=[log_name], max_scenes=max_scenes, num_history_frames=NUM_HISTORY,
        num_future_frames=NUM_FUTURE, frame_interval=FRAME_INTERVAL, has_route=True,
    )
    return filter_scenes(log_dir, scene_filter)[0]


def select_scenes(log_dir: Path, max_logs: int, per_log: int) -> dict:
    """Round-robin a bounded sample across three disjoint map locations."""
    if max_logs < 3 or per_log < 1:
        raise ValueError("map-disjoint pilot needs at least three logs and one scene per log")
    logs_by_location = defaultdict(list)
    for path in sorted(log_dir.glob("*.pkl")):
        if not path.is_file():
            continue
        log_name = _identifier(path.stem, "log")
        records = []
        for token, frames in sorted(_scene_frames(log_dir, log_name, per_log).items()):
            current = frames[NUM_HISTORY - 1]
            official_log_name = _identifier(current["log_name"], "log")
            if official_log_name != log_name:
                raise ValueError(f"log filename {log_name!r} does not match metadata {official_log_name!r}")
            location = _identifier(current["map_location"], "map location")
            if location == "las_vegas":
                location = "us-nv-las-vegas-strip"
            records.append({"token": _identifier(token, "token"), "log": log_name, "group": location})
        if not records:
            continue
        map_names = {record["group"] for record in records}
        if len(map_names) != 1:
            raise ValueError(f"log {log_name!r} spans multiple locations")
        logs_by_location[records[0]["group"]].append(records)

    selected_locations = sorted(logs_by_location, key=lambda name: (-len(logs_by_location[name]), name))[:3]
    if len(selected_locations) < 3:
        names = ", ".join(sorted(logs_by_location)) or "none"
        raise ValueError(f"need three map locations with eligible scenes for disjoint groups; found {names}")

    result = {split: [] for split in SPLITS}
    selected_logs = 0
    for log_idx in range(max(len(logs_by_location[name]) for name in selected_locations)):
        for map_idx, location in enumerate(selected_locations):
            if log_idx >= len(logs_by_location[location]):
                continue
            split = "evaluation" if map_idx == len(selected_locations) - 1 else "bank"
            result[split].extend(logs_by_location[location][log_idx])
            selected_logs += 1
            if selected_logs == max_logs:
                return result
    return result


def _validate_selection(selection: dict) -> None:
    seen = set()
    groups_by_split = {}
    logs_by_split = {}
    for split in SPLITS:
        groups_by_split[split] = set()
        logs_by_split[split] = set()
        for record in selection[split]:
            token = _identifier(record["token"], "token")
            location = _identifier(record["group"], "map location")
            log_name = _identifier(record["log"], "log")
            if token in seen:
                raise ValueError(f"duplicate scene token {token!r}")
            seen.add(token)
            groups_by_split[split].add(location)
            logs_by_split[split].add(log_name)
    if len(groups_by_split["bank"]) < 2 or not groups_by_split["evaluation"]:
        raise ValueError("selection needs at least two bank map locations and one held-out map location")
    if groups_by_split["bank"] & groups_by_split["evaluation"]:
        raise ValueError("bank and evaluation locations overlap")
    if logs_by_split["bank"] & logs_by_split["evaluation"]:
        raise ValueError("bank and evaluation logs overlap")


def resolve_manifest(cache_root: Path, selection: dict) -> tuple[dict, list[dict]]:
    _validate_selection(selection)
    result = {split: [] for split in SPLITS}
    missing = []
    for split in SPLITS:
        for record in selection[split]:
            directory = cache_root / record["log"]
            candidates = [
                path for path in directory.glob(f"*/{record['token']}/{CACHE_FILENAME}") if path.is_file()
            ]
            if not candidates:
                missing.append(record)
                continue
            if len(candidates) != 1:
                raise ValueError(f"multiple NAVSIM metric caches for token {record['token']!r}")
            result[split].append({**record, "cache": str(candidates[0].resolve())})
    return result, missing


def cache_command(log_dir: Path, cache_root: Path, records: list[dict]) -> list[str]:
    """Construct upstream Hydra arguments for exactly the requested scene tokens."""
    tokens = sorted({_identifier(row["token"], "token") for row in records})
    logs = sorted({_identifier(row["log"], "log") for row in records})
    if not tokens:
        raise ValueError("need at least one token to run metric caching")
    return [
        sys.executable, "-m", CACHE_MODULE, "train_test_split=mini", "gpu=false",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "force_feature_computation=false",
        f"navsim_log_path={json.dumps(str(log_dir), ensure_ascii=False)}",
        f"metric_cache_path={json.dumps(str(cache_root), ensure_ascii=False)}",
        f"train_test_split.scene_filter.log_names=[{','.join(logs)}]",
        f"train_test_split.scene_filter.tokens=[{','.join(tokens)}]",
        f"train_test_split.scene_filter.max_scenes={len(tokens)}",
        f"train_test_split.scene_filter.num_history_frames={NUM_HISTORY}",
        f"train_test_split.scene_filter.num_future_frames={NUM_FUTURE}",
        f"train_test_split.scene_filter.frame_interval={FRAME_INTERVAL}",
        "train_test_split.scene_filter.has_route=true",
    ]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, nargs="?", help="folder of trusted NAVSIM mini log pickles")
    parser.add_argument("cache_root", type=Path, nargs="?", help="NAVSIM metric cache output directory")
    parser.add_argument("--max-logs", type=int, default=8, help="maximum selected logs across both splits")
    parser.add_argument("--scenes-per-log", type=int, default=2, help="cap selected scenes per log")
    parser.add_argument("--manifest", type=Path, required=True, help="output for ct_sgf_navsim.py")
    parser.add_argument("--selection", type=Path, help="output selection JSON with reproducible scene tokens")
    parser.add_argument("--resume", type=Path, help="reuse a prior selection JSON instead of scanning the mini logs")
    parser.add_argument("--cache-missing", action="store_true", help="run official metric caching for missing tokens")
    args = parser.parse_args()
    if args.resume is None and (args.log_dir is None or args.cache_root is None):
        parser.error("provide log_dir and cache_root, or --resume a saved selection")
    if args.resume is not None and (args.log_dir is not None or args.cache_root is not None or args.selection is not None):
        parser.error("--resume already supplies log_dir, cache_root and the selection")

    try:
        if args.resume is not None:
            selection_path = args.resume.resolve()
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            log_dir = Path(selection["log_dir"]).resolve()
            cache_root = Path(selection["cache_root"]).resolve()
        else:
            selection_path = (args.selection or args.manifest.with_suffix(".selection.json")).resolve()
            log_dir = args.log_dir.resolve(strict=True)
            cache_root = args.cache_root.resolve()
            selection = {
                "log_dir": str(log_dir), "cache_root": str(cache_root),
                **select_scenes(log_dir, args.max_logs, args.scenes_per_log),
            }

        manifest_path = args.manifest.resolve()
        if selection_path == manifest_path:
            raise ValueError("selection and manifest paths must be different")
        manifest, missing = resolve_manifest(cache_root, selection)
        if args.resume is None:
            _write_json(selection_path, selection)
        print(f"Selection {selection_path}: {len(selection['bank'])} bank scenes and {len(selection['evaluation'])} held-out scenes.")

        if missing and args.cache_missing:
            absent = [key for key in ("NUPLAN_MAPS_ROOT", "OPENSCENE_DATA_ROOT") if not os.environ.get(key)]
            if absent:
                raise ValueError("set the official NAVSIM environment variables " + ", ".join(absent))
            subprocess.run(cache_command(log_dir, cache_root, missing), check=True)
            manifest, missing = resolve_manifest(cache_root, selection)
        if missing:
            retry = [sys.executable, __file__, "--resume", str(selection_path), "--manifest", str(manifest_path), "--cache-missing"]
            print(f"{len(missing)} selected scenes still need NAVSIM metric caching.", file=sys.stderr)
            print("Resume in the NAVSIM Linux environment:", shlex.join(retry), file=sys.stderr)
            parser.exit(2)
        _write_json(manifest_path, manifest)
        print(f"Cache manifest written to {manifest_path}.")
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
