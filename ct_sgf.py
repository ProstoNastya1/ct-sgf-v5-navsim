"""Kernel first-event estimation and bounded search for CT-SGF v5."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math


Vector = tuple[float, ...]
FeatureSequence = Sequence[Sequence[float]]


def _vector(values: Sequence[float]) -> Vector:
    result = tuple(values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("expected a nonempty vector of finite numbers")
    return result


@dataclass(frozen=True)
class Rollout:
    """A simulated plan with a known observation prefix and optional first event."""

    group: str
    features: tuple[Vector, ...]
    observed_steps: int
    first_event: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group.strip():
            raise ValueError("group must identify the rollout's log/location")
        features = tuple(_vector(row) for row in self.features)
        if not features or any(len(row) != len(features[0]) for row in features):
            raise ValueError("all transition feature vectors must share a dimension")
        object.__setattr__(self, "features", features)
        if type(self.observed_steps) is not int or not 0 <= self.observed_steps <= len(features):
            raise ValueError("observed_steps must count a prefix of features")
        if self.first_event is not None and (
            type(self.first_event) is not int or not 0 <= self.first_event < self.observed_steps
        ):
            raise ValueError("first_event must be a zero-based index in the observed prefix")

    def at_risk(self) -> Iterable[tuple[Vector, bool]]:
        """Retain the first-event step, or stop at censoring when no event is known."""
        stop = self.observed_steps if self.first_event is None else self.first_event + 1
        for step, features in enumerate(self.features[:stop]):
            yield features, step == self.first_event


@dataclass(frozen=True)
class StepEstimate:
    hazard: float
    exposure: float
    event_count: float
    effective_sample_size: float
    group_count: int
    supported: bool
    log_no_event: float


@dataclass(frozen=True)
class Prediction:
    """Smoothed scores remain available for diagnostics when support is insufficient."""

    steps: tuple[StepEstimate, ...]

    @property
    def supported(self) -> bool:
        return all(step.supported for step in self.steps)

    @property
    def log_survival(self) -> float:
        return math.fsum(step.log_no_event for step in self.steps)

    @property
    def failure_probability(self) -> float:
        return -math.expm1(self.log_survival)


class HazardModel:
    """Equation (3), with a product Gaussian kernel over transition features."""

    def __init__(
        self,
        rollouts: Iterable[Rollout],
        bandwidths: Sequence[float],
        *,
        alpha: float = 0.25,
        beta: float = 4.75,
        min_exposure: float = 1.0,
        min_ess: float = 2.0,
        min_group_exposure: float = 0.25,
        min_groups: int = 2,
        progress_gate: tuple[int, float] | None = None,
    ) -> None:
        self.bandwidths = _vector(bandwidths)
        settings = (alpha, beta, min_exposure, min_ess, min_group_exposure)
        if any(not math.isfinite(value) or value <= 0 for value in (*self.bandwidths, *settings)):
            raise ValueError("bandwidths, prior parameters and support thresholds must be positive")
        if not math.isfinite(alpha + beta):
            raise ValueError("prior strength must be finite")
        if type(min_groups) is not int or min_groups < 1:
            raise ValueError("min_groups must be a positive integer")
        self.alpha = alpha
        self.beta = beta
        self.min_exposure = min_exposure
        self.min_ess = min_ess
        self.min_group_exposure = min_group_exposure
        self.min_groups = min_groups
        if progress_gate is not None:
            index, tolerance = progress_gate
            if type(index) is not int or not 0 <= index < len(self.bandwidths):
                raise ValueError("progress feature index is out of range")
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("progress tolerance must be finite and positive")
        self.progress_gate = progress_gate
        self._groups: dict[str, list[tuple[Vector, bool]]] = {}
        for rollout in rollouts:
            if len(rollout.features[0]) != len(self.bandwidths):
                raise ValueError("bank feature dimension must equal the number of bandwidths")
            self._groups.setdefault(rollout.group, []).extend(rollout.at_risk())

    def _estimate(
        self, query: Vector, groups: Sequence[Sequence[tuple[Vector, bool]]]
    ) -> StepEstimate:
        event_weights = []
        safe_weights = []
        group_masses = []
        for group in groups:
            weights = []
            for stored, event in group:
                if self.progress_gate is not None:
                    index, tolerance = self.progress_gate
                    if abs(query[index] - stored[index]) > tolerance:
                        continue
                scaled_distance = math.hypot(
                    *((value - reference) / width
                      for value, reference, width in zip(query, stored, self.bandwidths))
                )
                weight = math.exp(-0.5 * scaled_distance * scaled_distance)
                weights.append(weight)
                (event_weights if event else safe_weights).append(weight)
            group_masses.append(math.fsum(weights))

        events = math.fsum(event_weights)
        safe = math.fsum(safe_weights)
        exposure = events + safe
        if exposure:
            # Normalize before squaring so tiny nonzero weights do not underflow.
            ess = 1.0 / math.fsum(
                (weight / exposure) ** 2 for weights in (event_weights, safe_weights)
                for weight in weights
            )
        else:
            ess = 0.0
        group_count = sum(mass >= self.min_group_exposure for mass in group_masses)
        supported = (
            exposure >= self.min_exposure
            and ess >= self.min_ess
            and group_count >= self.min_groups
        )
        event_mass = self.alpha + events
        safe_mass = self.beta + safe
        hazard = event_mass / (safe_mass + event_mass)
        odds = event_mass / safe_mass
        log_no_event = (
            -math.log1p(odds) if math.isfinite(odds)
            else math.log(safe_mass) - math.log(event_mass)
        )
        return StepEstimate(hazard, exposure, events, ess, group_count, supported, log_no_event)

    def predict(
        self, features: FeatureSequence, *, exclude_groups: Iterable[str] = ()
    ) -> Prediction:
        queries = tuple(_vector(row) for row in features)
        if not queries or any(len(row) != len(self.bandwidths) for row in queries):
            raise ValueError("expected one or more query vectors with the bank feature dimension")
        excluded = set(exclude_groups)
        groups = [rows for group, rows in self._groups.items() if group not in excluded]
        return Prediction(tuple(self._estimate(query, groups) for query in queries))


@dataclass(frozen=True)
class CandidateEvaluation:
    controls: Vector
    distance: float
    feasible: bool
    prediction: Prediction | None
    objective: float | None


@dataclass(frozen=True)
class Correction:
    selected: Vector
    base_prediction: Prediction
    selected_prediction: Prediction
    reason: str
    evaluations: tuple[CandidateEvaluation, ...]


def correct_plan(
    base: Sequence[float],
    proposals: Iterable[Sequence[float]],
    *,
    features_for: Callable[[Vector], FeatureSequence],
    model: HazardModel,
    lower: Sequence[float],
    upper: Sequence[float],
    control_scale: Sequence[float],
    max_distance: float,
    eta: float = 1.0,
    min_gain: float = 0.0,
    exclude_groups: Iterable[str] = (),
) -> Correction:
    """Evaluate Eq. (5) over one fixed candidate set with diagonal control covariance."""
    original = _vector(base)
    lower, upper, control_scale = map(_vector, (lower, upper, control_scale))
    if any(len(row) != len(original) for row in (lower, upper, control_scale)):
        raise ValueError("control bounds and scales must have the base plan dimension")
    if any(lo > hi or scale <= 0 for lo, hi, scale in zip(lower, upper, control_scale)):
        raise ValueError("each control needs ordered bounds and a positive scale")
    if any(not math.isfinite(value) or value < 0 for value in (max_distance, eta, min_gain)):
        raise ValueError("max_distance, eta and min_gain must be finite and nonnegative")
    if any(not lo <= value <= hi for value, lo, hi in zip(original, lower, upper)):
        raise ValueError("the base plan must satisfy control bounds")

    excluded = tuple(exclude_groups)
    base_prediction = model.predict(features_for(original), exclude_groups=excluded)
    base_value = -eta * base_prediction.log_survival if base_prediction.supported else None
    baseline = CandidateEvaluation(original, 0.0, True, base_prediction, base_value)
    evaluations = [baseline]
    seen = {original}
    selected = baseline
    for proposal in proposals:
        controls = _vector(proposal)
        if len(controls) != len(original):
            raise ValueError("every candidate must have the base plan dimension")
        if controls in seen:
            continue
        seen.add(controls)
        distance = math.hypot(
            *((value - ref) / scale for value, ref, scale in zip(controls, original, control_scale))
        )
        feasible = distance <= max_distance and all(
            lo <= value <= hi for value, lo, hi in zip(controls, lower, upper)
        )
        prediction = None
        objective = None
        if feasible:
            prediction = model.predict(features_for(controls), exclude_groups=excluded)
            if len(prediction.steps) != len(base_prediction.steps):
                raise ValueError("all candidates must be evaluated over the same prediction horizon")
            if prediction.supported:
                objective = 0.5 * distance * distance - eta * prediction.log_survival
        evaluation = CandidateEvaluation(controls, distance, feasible, prediction, objective)
        evaluations.append(evaluation)
        if objective is not None and selected.objective is not None and objective < selected.objective:
            selected = evaluation

    if base_value is None:
        reason = "base_unsupported"
    elif selected.objective is None or base_value - selected.objective <= min_gain:
        reason = "no_supported_improvement"
        selected = baseline
    else:
        reason = "improved"
    assert selected.prediction is not None
    return Correction(selected.controls, base_prediction, selected.prediction, reason, tuple(evaluations))
