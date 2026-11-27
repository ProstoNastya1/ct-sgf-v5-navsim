"""CT-SGF feature and label adapter for the pinned NAVSIM v2 PDM scorer.

Uses NAVSIM commit 0a380a9063d7162ec93d0f51e9990ebac585f720 and its
four-second, 10 Hz replay. The cache supplies the current ego state, route,
and time-zero annotated object tracks. Those tracks are an oracle-perception
input for this pilot. Logged future tracks are used solely by offline scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
import lzma
import math
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.metric_caching.metric_cache import MetricCache


NAVSIM_COMMIT = "0a380a9063d7162ec93d0f51e9990ebac585f720"
EVENT_TYPE = "first_at_fault_collision"

FEATURE_NAMES = (
    "time_s", "entry_x_m", "entry_y_m", "entry_heading_sin", "entry_heading_cos",
    "entry_speed_mps", "entry_accel_mps2", "entry_yaw_rate_radps", "route_progress_m",
    "route_offset_m", "preview_dx_m", "preview_dy_m", "nearby_track_present",
    "track_dx_m", "track_dy_m", "track_closing_speed_mps",
)
BANDWIDTHS = (0.20, 6.0, 2.0, 0.25, 0.25, 2.5, 1.5, 0.35, 6.0, 1.5, 4.0, 1.5, 0.3, 6.0, 3.0, 2.5)
CONTEXT_FEATURE_NAMES = (
    "current_ego_speed_mps", "current_route_offset_m", "current_track_present",
    "current_track_dx_m", "current_track_dy_m", "current_track_closing_speed_mps",
)
CONTEXT_BANDWIDTHS = (3.0, 2.0, 0.5, 10.0, 5.0, 3.0)
SCORE_COLUMNS = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
    "lane_keeping", "history_comfort", "pdm_score",
)

FeatureRows = tuple[tuple[float, ...], ...]
Track = tuple[float, float, float, float]
Pose = tuple[float, float, float]


@dataclass(frozen=True)
class NavsimAssessment:
    features: FeatureRows
    first_event: int | None
    observed_steps: int
    time_to_at_fault_collision: float | None
    score_metrics: dict[str, float | None]


@dataclass(frozen=True)
class NavsimCandidate:
    """Inference-time features and attained limits over future tracked states."""

    features: FeatureRows
    accel_min: float
    accel_max: float
    max_abs_steering_rate: float
    max_abs_speed: float


class InitialCollisionError(ValueError):
    """An at-fault collision is already ongoing before the forecast begins."""


def _to_ego_frame(dx: float, dy: float, heading: float) -> tuple[float, float]:
    cosine, sine = math.cos(heading), math.sin(heading)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _first_event(time_s: float, interval_length: float, horizon: int) -> int | None:
    if time_s == math.inf:
        return None
    if not math.isfinite(time_s):
        raise ValueError("NAVSIM returned an invalid at-fault collision time")
    step = round(time_s / interval_length)
    if not math.isclose(time_s, step * interval_length, abs_tol=1e-6):
        raise ValueError("NAVSIM collision time does not lie on the trajectory sampling grid")
    if step == 0:
        raise InitialCollisionError("at-fault collision precedes the forecast")
    if not 1 <= step <= horizon:
        raise ValueError("NAVSIM collision is outside the forecast transitions")
    # Official state index 1 is the end of the first predicted transition.
    return step - 1


def _nearest_track(
    tracks: tuple[Track, ...], x: float, y: float, heading: float, speed: float, entry_time: float
) -> tuple[float, float, float, float]:
    nearest_distance = 35.0
    nearest = (0.0, 0.0, 0.0, 0.0)
    for track_x, track_y, velocity_x, velocity_y in tracks:
        dx = track_x + entry_time * velocity_x - x
        dy = track_y + entry_time * velocity_y - y
        distance = math.hypot(dx, dy)
        if distance < nearest_distance:
            nearest_distance = distance
            local_x, local_y = _to_ego_frame(dx, dy, heading)
            velocity_forward, _ = _to_ego_frame(velocity_x, velocity_y, heading)
            nearest = (1.0, local_x, local_y, speed - velocity_forward)
    return nearest


class NavsimAdapter:
    """Score NAVSIM v2 candidate trajectories with raw at-fault labels."""

    horizon = 40
    interval_length = 0.1
    time_horizon = 4.0
    feature_names = FEATURE_NAMES
    bandwidths = BANDWIDTHS
    context_feature_names = CONTEXT_FEATURE_NAMES
    context_bandwidths = CONTEXT_BANDWIDTHS
    progress_feature = "route_progress_m"
    event_type = EVENT_TYPE
    commit = NAVSIM_COMMIT

    def __init__(self, *, traffic: str = "non-reactive") -> None:
        if traffic not in ("non-reactive", "reactive"):
            raise ValueError("traffic must be non-reactive log replay or official reactive IDM")
        try:
            import numpy as np
            from shapely import Point
            from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            from navsim.common.dataclasses import Trajectory
            from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
            from navsim.planning.metric_caching.metric_cache import MetricCache
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
            from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
            from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
                state_array_to_coords_array,
            )
            from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex
            from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
        except ImportError as error:
            raise RuntimeError(
                f"Install NAVSIM {NAVSIM_COMMIT} and its dependencies before using this adapter: {error}"
            ) from error

        self._np = np
        self._point = Point
        self._ego_track_type = TrackedObjectType.EGO
        self._sampling_cls = TrajectorySampling
        self._trajectory_cls = Trajectory
        self._cache_cls = MetricCache
        self._get_trajectory_as_array = get_trajectory_as_array
        self._pdm_score = pdm_score
        self._transform_trajectory = transform_trajectory
        self._state_to_coords = state_array_to_coords_array
        self._center_index = BBCoordsIndex.CENTER
        self._state_index = StateIndex
        self._sampling = TrajectorySampling(num_poses=self.horizon, interval_length=self.interval_length)
        self._simulator = PDMSimulator(self._sampling)
        # The upstream human penalty filter makes an additional scoring pass and
        # can override collision metrics. We need the raw candidate event and scores.
        self._scorer = PDMScorer(self._sampling, PDMScorerConfig(human_penalty_filter=False))
        if traffic == "reactive":
            from hydra import compose, initialize_config_module
            from hydra.utils import instantiate

            with initialize_config_module(
                config_module="navsim.planning.script.config.pdm_scoring", version_base=None
            ):
                config = compose(
                    "default_run_pdm_score_from_submission",
                    overrides=["train_test_split=navhard_two_stage"],
                )
            self._traffic_policy = instantiate(config.traffic_agents_policy.reactive, self._sampling)
        else:
            self._traffic_policy = LogReplayTrafficAgents(self._sampling)

    def load_cache(self, path: str | Path) -> MetricCache:
        """Read a trusted official metric_cache.pkl, the same lzma/pickle format as NAVSIM."""
        with lzma.open(path, "rb") as stream:
            metric_cache = pickle.load(stream)
        if not isinstance(metric_cache, self._cache_cls):
            raise ValueError(f"{path} does not contain a NAVSIM v2 MetricCache")
        return metric_cache

    def initial_speed(self, metric_cache: MetricCache) -> float:
        velocity = metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d
        return math.hypot(velocity.x, velocity.y)

    def _waypoint_sampling(self, waypoint_interval: float):
        if not math.isfinite(waypoint_interval) or waypoint_interval <= 0:
            raise ValueError("waypoint_interval must be finite and positive")
        num_poses = round(self.time_horizon / waypoint_interval)
        if num_poses < 2 or not math.isclose(num_poses * waypoint_interval, self.time_horizon):
            raise ValueError("the waypoint interval must divide the four-second forecast into at least two steps")
        return self._sampling_cls(num_poses=num_poses, interval_length=waypoint_interval)

    def straight_poses(
        self, metric_cache: MetricCache, *, acceleration: float = 0.0, waypoint_interval: float = 0.5
    ) -> tuple[Pose, ...]:
        """Generate a straight pilot proposal; clamp speed to zero after braking to a stop."""
        if not math.isfinite(acceleration):
            raise ValueError("acceleration must be finite")
        speed = self.initial_speed(metric_cache)
        sampling = self._waypoint_sampling(waypoint_interval)
        stop_time = -speed / acceleration if acceleration < 0 else self.time_horizon
        points = []
        for step in range(1, sampling.num_poses + 1):
            time = min(step * waypoint_interval, stop_time)
            distance = speed * time + 0.5 * acceleration * time * time
            points.append((distance, 0.0, 0.0))
        return tuple(points)

    def pdm_base_poses(
        self, metric_cache: MetricCache, *, waypoint_interval: float = 0.5
    ) -> tuple[Pose, ...]:
        """Extract the cached PDM-Closed proposal, computed from time-zero route/history."""
        ego = metric_cache.ego_state
        origin = ego.rear_axle
        sampling = self._waypoint_sampling(waypoint_interval)
        states = self._get_trajectory_as_array(metric_cache.trajectory, sampling, ego.time_point)
        index = self._state_index
        poses = []
        for state in states[1:]:
            x, y = _to_ego_frame(float(state[index.X]) - origin.x, float(state[index.Y]) - origin.y, origin.heading)
            angle = float(state[index.HEADING]) - origin.heading
            poses.append((x, y, math.atan2(math.sin(angle), math.cos(angle))))
        return tuple(poses)

    def _trajectory(self, poses: Sequence[Sequence[float]], waypoint_interval: float) -> Trajectory:
        sampling = self._waypoint_sampling(waypoint_interval)
        values = self._np.asarray(poses, dtype=self._np.float64)
        if values.shape != (sampling.num_poses, 3) or not self._np.isfinite(values).all():
            raise ValueError(f"expected {sampling.num_poses} finite ego-relative [x, y, heading] waypoints")
        return self._trajectory_cls(poses=values, trajectory_sampling=sampling)

    def _reference_states(self, metric_cache: MetricCache, trajectory: Trajectory):
        ego = metric_cache.ego_state
        global_trajectory = self._transform_trajectory(trajectory, ego)
        return self._get_trajectory_as_array(global_trajectory, self._sampling, ego.time_point)

    def _current_tracks(self, metric_cache: MetricCache) -> tuple[Track, ...]:
        tracks = []
        for detections in metric_cache.current_tracked_objects:
            for track in detections.tracked_objects:
                if track.tracked_object_type == self._ego_track_type:
                    continue
                center = track.box.center
                velocity = getattr(track, "velocity", None)
                values = (
                    float(center.x), float(center.y),
                    float(velocity.x) if velocity is not None else 0.0,
                    float(velocity.y) if velocity is not None else 0.0,
                )
                if all(math.isfinite(value) for value in values):
                    tracks.append(values)
        return tuple(tracks)

    def scene_context(self, metric_cache: MetricCache) -> tuple[float, ...]:
        """Retrieve using features independent of the candidate and its future labels."""
        ego = metric_cache.ego_state
        origin = ego.rear_axle
        speed = self.initial_speed(metric_cache)
        route_offset = self._point(origin.x, origin.y).distance(metric_cache.centerline.linestring)
        return (
            speed, float(route_offset),
            *_nearest_track(self._current_tracks(metric_cache), origin.x, origin.y, origin.heading, speed, 0.0),
        )

    def _transition_features(self, metric_cache: MetricCache, simulated, reference) -> FeatureRows:
        index = self._state_index
        ego = metric_cache.ego_state.rear_axle
        tracks = self._current_tracks(metric_cache)
        centerline = metric_cache.centerline
        points = self._point
        first_progress = float(centerline.project(points(ego.x, ego.y)))
        preview_steps = round(1.0 / self.interval_length)
        features = []
        for step in range(self.horizon):
            x = float(simulated[step, index.X])
            y = float(simulated[step, index.Y])
            heading = float(simulated[step, index.HEADING])
            speed = math.hypot(simulated[step, index.VELOCITY_X], simulated[step, index.VELOCITY_Y])
            entry_x, entry_y = _to_ego_frame(x - ego.x, y - ego.y, ego.heading)
            heading_delta = heading - ego.heading
            point = points(x, y)
            preview_idx = min(step + preview_steps, self.horizon)
            preview_dx, preview_dy = _to_ego_frame(
                float(reference[preview_idx, index.X] - reference[step, index.X]),
                float(reference[preview_idx, index.Y] - reference[step, index.Y]),
                heading,
            )
            features.append((
                (step + 1) * self.interval_length,
                entry_x, entry_y, math.sin(heading_delta), math.cos(heading_delta),
                speed, float(simulated[step, index.ACCELERATION_X]),
                float(simulated[step, index.ANGULAR_VELOCITY]),
                float(centerline.project(point)) - first_progress,
                float(point.distance(centerline.linestring)),
                preview_dx, preview_dy,
                *_nearest_track(tracks, x, y, heading, speed, step * self.interval_length),
            ))
        return tuple(features)

    def features_for(
        self, metric_cache: MetricCache, poses: Sequence[Sequence[float]], *, waypoint_interval: float = 0.5
    ) -> FeatureRows:
        return self.tracked_candidate(metric_cache, poses, waypoint_interval=waypoint_interval).features

    def tracked_candidate(
        self, metric_cache: MetricCache, poses: Sequence[Sequence[float]], *, waypoint_interval: float = 0.5
    ) -> NavsimCandidate:
        """Run the official tracker for a candidate without reading future observation data."""
        reference = self._reference_states(metric_cache, self._trajectory(poses, waypoint_interval))
        simulated = self._simulator.simulate_proposals(reference[None, ...], metric_cache.ego_state)[0]
        if not self._np.isfinite(simulated).all():
            raise ValueError("NAVSIM tracker produced non-finite candidate states")
        index = self._state_index
        # Initial conditions are already observed. Include the final future state
        # here, even though transition features reference only each step's entry.
        future = simulated[1:]
        accel = future[:, index.ACCELERATION_X]
        speed = self._np.hypot(future[:, index.VELOCITY_X], future[:, index.VELOCITY_Y])
        return NavsimCandidate(
            self._transition_features(metric_cache, simulated, reference),
            float(accel.min()),
            float(accel.max()),
            float(self._np.abs(future[:, index.STEERING_RATE]).max()),
            float(speed.max()),
        )

    def assess(
        self, metric_cache: MetricCache, poses: Sequence[Sequence[float]], *, waypoint_interval: float = 0.5
    ) -> NavsimAssessment:
        """Read replay data to compute the offline label and raw single-stage metrics."""
        trajectory = self._trajectory(poses, waypoint_interval)
        # score_proposals replaces these fields on the observation. Give it a fresh
        # copy per candidate so subsequent replay starts from the original cache.
        scoring_cache = replace(metric_cache, observation=copy(metric_cache.observation))
        score, simulated = self._pdm_score(
            metric_cache=scoring_cache, model_trajectory=trajectory, future_sampling=self._sampling,
            simulator=self._simulator, scorer=self._scorer, traffic_agents_policy=self._traffic_policy,
        )
        collision_time = float(self._scorer.time_to_at_fault_collision(1))
        first_event = _first_event(collision_time, self.interval_length, self.horizon)
        raw_metrics = {column: float(score[column].iat[0]) for column in SCORE_COLUMNS}
        collision_metric = raw_metrics["no_at_fault_collisions"]
        if not math.isfinite(collision_metric) or (collision_metric < 1.0) != (first_event is not None):
            raise RuntimeError("NAVSIM collision metric and first-event label disagree")
        metrics = {
            column: value if math.isfinite(value) else None for column, value in raw_metrics.items()
        }
        coordinates = self._state_to_coords(
            simulated[None, ...], metric_cache.ego_state.car_footprint.vehicle_parameters
        )
        centers = coordinates[0, [0, -1], self._center_index]
        progress_start, progress_end = metric_cache.centerline.project([self._point(*center) for center in centers])
        metrics["ego_progress_m"] = max(0.0, float(progress_end - progress_start))
        reference = self._reference_states(metric_cache, trajectory)
        return NavsimAssessment(
            self._transition_features(metric_cache, simulated, reference), first_event, self.horizon,
            collision_time if first_event is not None else None, metrics,
        )


class EgoOnlyNavsimAdapter(NavsimAdapter):
    """Use the tracked candidate and observed ego dynamics for inference features."""

    feature_names = (
        "time_s", "entry_x_m", "entry_y_m", "entry_heading_sin", "entry_heading_cos",
        "entry_speed_mps", "entry_accel_mps2", "path_progress_m", "preview_dx_m", "preview_dy_m",
    )
    bandwidths = (0.20, 6.0, 2.0, 0.25, 0.25, 2.5, 1.5, 6.0, 4.0, 1.5)
    context_feature_names = ("current_ego_speed_mps", "current_ego_accel_mps2")
    context_bandwidths = (3.0, 1.5)
    progress_feature = "path_progress_m"

    def scene_context(self, metric_cache: MetricCache) -> tuple[float, ...]:
        dynamic = metric_cache.ego_state.dynamic_car_state
        return (self.initial_speed(metric_cache), float(dynamic.rear_axle_acceleration_2d.x))

    def _transition_features(self, metric_cache: MetricCache, simulated, reference) -> FeatureRows:
        index = self._state_index
        origin = metric_cache.ego_state.rear_axle
        preview_steps = round(1.0 / self.interval_length)
        rows = []
        progress = 0.0
        previous_x, previous_y = origin.x, origin.y
        for step in range(self.horizon):
            x = float(simulated[step, index.X])
            y = float(simulated[step, index.Y])
            heading = float(simulated[step, index.HEADING])
            progress += math.hypot(x - previous_x, y - previous_y)
            previous_x, previous_y = x, y
            entry = _to_ego_frame(x - origin.x, y - origin.y, origin.heading)
            delta = heading - origin.heading
            speed = math.hypot(simulated[step, index.VELOCITY_X], simulated[step, index.VELOCITY_Y])
            preview_idx = min(step + preview_steps, self.horizon)
            preview = _to_ego_frame(
                float(reference[preview_idx, index.X] - reference[step, index.X]),
                float(reference[preview_idx, index.Y] - reference[step, index.Y]),
                heading,
            )
            rows.append((
                (step + 1) * self.interval_length, *entry, math.sin(delta), math.cos(delta), speed,
                float(simulated[step, index.ACCELERATION_X]), progress, *preview,
            ))
        return tuple(rows)
