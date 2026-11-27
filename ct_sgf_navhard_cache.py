"""Cache every official navhard token using NAVSIM's pinned MetricCacheProcessor."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
import os
import multiprocessing
from pathlib import Path
import pickle
import threading

from ct_sgf_navhard_score import official_tokens


_fork_cache_token = None


def _run_forked_cache_token(token, is_synthetic):
    return _fork_cache_token(token, is_synthetic)


def create_cache(root, workers, limit, processes=False):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.training.experiments.cache_metadata_entry import save_cache_metadata

    with initialize_config_module(config_module="navsim.planning.script.config.metric_caching", version_base=None):
        cfg = compose("default_metric_caching", overrides=["train_test_split=navhard_two_stage"])
    originals, synthetic = official_tokens()
    if limit:
        originals, synthetic = originals[:limit], synthetic[:limit]
    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    sensor_config = SensorConfig.build_no_sensors()
    scene_filter = replace(instantiate(cfg.train_test_split.scene_filter), include_synthetic_scenes=False,
                           tokens=list(originals), synthetic_scene_tokens=None)
    loader = SceneLoader(
        data_path=data / "navsim_logs/test", original_sensor_path=None,
        scene_filter=scene_filter, sensor_config=sensor_config,
    )
    if set(loader.tokens_stage_one) != set(originals):
        raise ValueError("original logs do not supply every required navhard token")
    synthetic_folder = data / "navhard_two_stage/synthetic_scene_pickles"
    # The filename can differ from the official initial token inside the scene.
    remaining = set(synthetic)
    available = {}
    for file in synthetic_folder.glob("*.pkl"):
        with file.open("rb") as stream:
            token = pickle.load(stream)["scene_metadata"]["initial_token"]
        if token in remaining:
            available[token] = file
            remaining.remove(token)
        if not remaining:
            break
    if set(synthetic) - available.keys():
        raise ValueError("synthetic archive does not supply every required navhard token")
    local = threading.local()

    def cache_token(token, is_synthetic):
        if not hasattr(local, "processor"):
            local.processor = MetricCacheProcessor(
                cache_path=str(root), force_feature_computation=False,
                proposal_sampling=instantiate(cfg.proposal_sampling),
            )
        scene = (
            Scene.load_from_disk(available[token], None, sensor_config)
            if is_synthetic else loader.get_scene_from_token(token)
        )
        scenario = NavSimScenario(
            scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0",
        )
        entry = local.processor.compute_and_save_metric_cache(scenario)
        if entry is None:
            raise RuntimeError(f"NAVSIM did not cache token {token}")
        return entry

    cache_one = cache_token
    if processes:
        # After fork each worker owns its simulator, without duplicating the
        # serial pass over the arbitrary raw pickle filenames.
        global _fork_cache_token
        _fork_cache_token = cache_token
        cache_one = _run_forked_cache_token
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
    root.mkdir(parents=True, exist_ok=True)
    metadata = []
    with executor:
        tasks = [executor.submit(cache_one, token, flag)
                 for stage, flag in ((originals, False), (synthetic, True)) for token in stage]
        for index, task in enumerate(as_completed(tasks), start=1):
            metadata.append(task.result())
            if index % 100 == 0 or index == len(tasks):
                print(f"Official metric caches {index}/{len(tasks)}", flush=True)
    save_cache_metadata(metadata, root, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-stage", type=int, default=0)
    parser.add_argument("--processes", action="store_true", help="use forked workers on Linux")
    args = parser.parse_args()
    if args.workers < 1 or args.limit_per_stage < 0:
        parser.error("worker count must be positive and the limit nonnegative")
    create_cache(args.cache.resolve(), args.workers, args.limit_per_stage, args.processes)


if __name__ == "__main__":
    main()
