"""Run a small, deterministic CT-SGF v5 example in a toy driving simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import exp

from ct_sgf import HazardModel, Rollout, correct_plan
from ct_sgf_baselines import EventOnlyModel


DT = 0.5
BASE = (0.0,) * 8
EVENT_STEP = len(BASE) - 1
HOLDOUT_GROUP = "demo-location"


@dataclass(frozen=True)
class Road:
    start: float = 0.0
    speed: float = 4.0
    gate: float = 14.5
    speed_limit: float = 3.1


@dataclass(frozen=True)
class Simulation:
    features: tuple[tuple[float, ...], ...]
    first_event: int | None
    crossed_gate: bool


def simulate(road: Road, controls: tuple[float, ...]) -> Simulation:
    """Flag a gate-crossing interval whose starting speed is above the limit."""
    position = road.start
    speed = road.speed
    features = []
    first_event = None
    crossed_gate = False
    for index, acceleration in enumerate(controls):
        features.append(((index + 1) * DT, position - road.gate, speed, acceleration))
        next_speed = speed + acceleration * DT
        if next_speed < 0:
            raise ValueError("the toy simulator only models forward motion")
        next_position = position + (speed + next_speed) * DT / 2
        crosses_gate = position < road.gate <= next_position
        crossed_gate = crossed_gate or crosses_gate
        if first_event is None and crosses_gate and speed > road.speed_limit:
            first_event = index
        position, speed = next_position, next_speed
    return Simulation(tuple(features), first_event, crossed_gate)


def training_bank() -> tuple[Rollout, ...]:
    """Collect first-event and exposure records from separate toy locations."""
    records = []
    for location in range(8):
        group = f"bank-location-{location}"
        start = 50.0 * location
        road = Road(
            start=start,
            speed=3.93 + 0.02 * location,
            gate=start + 14.48 + 0.01 * (location % 4),
        )
        pulse = 1.35 + 0.06 * location
        remote_prefix = (pulse, -2 * pulse, pulse, 0.0, 0.0, 0.0, 0.0, 0.0)
        slow_entry = (0.0, 0.0, 0.0, 1.8, -1.8, 0.0, -3.8, 0.0)
        braking_grid = tuple(
            BASE[:6] + (braking, 0.0)
            for braking in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
        )
        for controls in (*braking_grid, remote_prefix, slow_entry):
            outcome = simulate(road, controls)
            records.append(Rollout(group, outcome.features, len(controls), outcome.first_event))

        # A censored recording supplies exposure only for its observed prefix.
        censored = simulate(road, BASE)
        records.append(Rollout(group, censored.features, observed_steps=5))

    # The event is before the end, so later simulated states are not exposures.
    early = simulate(Road(speed=4.5), BASE)
    records.append(Rollout("early-event-location", early.features, len(BASE), early.first_event))
    return tuple(records)


def gaussian_affinity(
    first: tuple[float, ...], second: tuple[float, ...], bandwidth: float
) -> float:
    distance_squared = sum(((a - b) / bandwidth) ** 2 for a, b in zip(first, second))
    return exp(-0.5 * distance_squared)


def event_only_features(simulation: Simulation) -> tuple[float, ...]:
    time, position_to_gate, _, acceleration = simulation.features[EVENT_STEP]
    return time, position_to_gate, acceleration


def run_demo() -> dict:
    """Return the constructed examples and fixed-candidate correction results."""
    road = Road()
    bank = training_bank()
    model = HazardModel(
        bank,
        bandwidths=(0.19, 0.65, 0.36, 1.0),
        min_exposure=1.5,
        min_ess=2.0,
        min_group_exposure=0.15,
        min_groups=2,
    )
    event_only = EventOnlyModel(
        bank, model.bandwidths, (0, 1, 3), min_exposure=1.5, min_ess=2.0,
        min_group_exposure=0.15, min_groups=2,
    )
    excluded = (HOLDOUT_GROUP,)
    remote_controls = (2.0, -4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    slow_controls = (0.0, 0.0, 0.0, 2.0, -2.0, 0.0, -4.0, 0.0)
    base_simulation = simulate(road, BASE)
    remote = simulate(road, remote_controls)
    slow = simulate(road, slow_controls)
    base_prediction = model.predict(base_simulation.features, exclude_groups=excluded)
    remote_prediction = model.predict(remote.features, exclude_groups=excluded)
    slow_prediction = model.predict(slow.features, exclude_groups=excluded)

    proposals = tuple(
        BASE[:6] + (acceleration, 0.0)
        for acceleration in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 4.5)
    )
    # This early pulse is inside the control bounds but far from the memory.
    proposals += ((-2.5, 2.5, *BASE[2:]),)
    correction = correct_plan(
        BASE,
        proposals,
        features_for=lambda controls: simulate(road, controls).features,
        model=model,
        lower=(-4.0,) * len(BASE),
        upper=(4.0,) * len(BASE),
        control_scale=(2.5,) * len(BASE),
        max_distance=1.6,
        eta=5.0,
        min_gain=0.02,
        exclude_groups=excluded,
    )

    # Rank exactly the same physically feasible controls by the simulator label.
    # Distance only breaks ties between plans with the same first-event outcome.
    oracle_pool = [
        (simulate(road, candidate.controls), candidate)
        for candidate in correction.evaluations
        if candidate.feasible
    ]
    oracle_simulation, oracle = min(
        oracle_pool,
        key=lambda result: (result[0].first_event is not None, result[1].distance),
    )
    after = simulate(road, correction.selected)
    base_step = base_prediction.steps[EVENT_STEP]
    remote_step = remote_prediction.steps[EVENT_STEP]
    slow_step = slow_prediction.steps[EVENT_STEP]
    projected_hazards = [
        event_only.predict(simulation.features, exclude_groups=excluded).steps[EVENT_STEP].hazard
        for simulation in (base_simulation, slow)
    ]
    predictions = (correction.base_prediction, correction.selected_prediction)

    return {
        "scope": "Constructed simulator examples for checking mechanics",
        "step_duration_s": DT,
        "bank": {
            "rollouts": len(bank),
            "groups": len({rollout.group for rollout in bank}),
            "observed_first_events": sum(rollout.first_event is not None for rollout in bank),
            "at_risk_steps": sum(1 for rollout in bank for _ in rollout.at_risk()),
        },
        "remote_prefix_pair": {
            "first_events_zero_based": [base_simulation.first_event, remote.first_event],
            "same_entry_features": base_simulation.features[EVENT_STEP] == remote.features[EVENT_STEP],
            "whole_control_chunk_raw_kernel": gaussian_affinity(BASE, remote_controls, 1.0),
            "conditional_hazards": [base_step.hazard, remote_step.hazard],
            "both_event_steps_supported": base_step.supported and remote_step.supported,
        },
        "entry_speed_pair": {
            "first_events_zero_based": [base_simulation.first_event, slow.first_event],
            "both_crossed_gate": base_simulation.crossed_gate and slow.crossed_gate,
            "entry_speeds_m_per_s": [
                base_simulation.features[EVENT_STEP][2], slow.features[EVENT_STEP][2]
            ],
            "event_position_action_raw_kernel": gaussian_affinity(
                event_only_features(base_simulation), event_only_features(slow), 1.0
            ),
            "conditional_hazards": [base_step.hazard, slow_step.hazard],
            "event_only_hazards": projected_hazards,
            "both_event_steps_supported": base_step.supported and slow_step.supported,
        },
        "correction": {
            "reason": correction.reason,
            "base_controls": BASE,
            "selected_controls": correction.selected,
            "first_events_zero_based": [base_simulation.first_event, after.first_event],
            "crossed_gate": [base_simulation.crossed_gate, after.crossed_gate],
            "predicted_failure_probabilities": [item.failure_probability for item in predictions],
            "supported_steps": [sum(step.supported for step in item.steps) for item in predictions],
            "evaluated_candidates": len(correction.evaluations),
            "feasible_candidates": len(oracle_pool),
            "supported_feasible_candidates": sum(candidate.objective is not None for _, candidate in oracle_pool),
            "oracle_first_event_zero_based": oracle_simulation.first_event,
            "oracle_crossed_gate": oracle_simulation.crossed_gate,
            "oracle_controls": oracle.controls,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()
    result = run_demo()
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("CT-SGF v5 synthetic example")
    print(result["scope"])
    bank = result["bank"]
    print(f"Bank has {bank['rollouts']} toy rollouts and {bank['at_risk_steps']} at-risk steps.")
    remote = result["remote_prefix_pair"]
    hazard_a, hazard_b = remote["conditional_hazards"]
    print(
        f"Remote-prefix pair, event steps {remote['first_events_zero_based']}, raw chunk affinity "
        f"{remote['whole_control_chunk_raw_kernel']:.3g}, "
        f"conditional hazards {hazard_a:.3f} and {hazard_b:.3f}."
    )
    speed = result["entry_speed_pair"]
    hazard_a, hazard_b = speed["conditional_hazards"]
    print(
        f"Different entry speeds, event steps {speed['first_events_zero_based']}, position/action affinity "
        f"{speed['event_position_action_raw_kernel']:.3f}, "
        f"conditional hazards {hazard_a:.3f} and {hazard_b:.3f}."
    )
    first, second = speed["event_only_hazards"]
    print(f"With entry speed removed, the matched steps score {first:.3f} and {second:.3f}.")
    correction = result["correction"]
    before, after = correction["predicted_failure_probabilities"]
    before_event, after_event = correction["first_events_zero_based"]
    print(f"Correction decision {correction['reason']}, estimated failure probability {before:.3f} to {after:.3f}.")
    print(f"Simulator first event {before_event} to {after_event} (zero-based, None means no event).")
    print(f"Gate crossed before and after correction {correction['crossed_gate']}.")
    print(
        f"Oracle first event {correction['oracle_first_event_zero_based']} across "
        f"{correction['feasible_candidates']} physically feasible candidates, "
        f"of which {correction['supported_feasible_candidates']} had sufficient bank support."
    )
    print(f"Oracle crossed the gate {correction['oracle_crossed_gate']}.")


if __name__ == "__main__":
    main()
