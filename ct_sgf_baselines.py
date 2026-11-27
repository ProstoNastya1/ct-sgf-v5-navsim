"""Held-out trajectory and event baselines on the CT-SGF evidence bank."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from ct_sgf import FeatureSequence, HazardModel, Prediction, Rollout, Vector


@dataclass(frozen=True)
class Risk:
    """A whole-trajectory score with stable logs for both outcome classes."""

    failure_probability: float
    log_survival: float
    log_failure: float

    @classmethod
    def from_logit(cls, logit: float) -> Risk:
        softplus = math.log1p(math.exp(-abs(logit)))
        log_failure = min(0.0, logit) - softplus
        log_survival = min(0.0, -logit) - softplus
        return cls(math.exp(log_failure), log_survival, log_failure)


def _flatten(features: FeatureSequence, bandwidths: Vector, horizon: int) -> Vector:
    if len(features) != horizon or any(len(row) != len(bandwidths) for row in features):
        raise ValueError("trajectory baselines require the full horizon and bank feature dimension")
    values = tuple(value / width for row in features for value, width in zip(row, bandwidths))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("trajectory baseline features must be finite after bandwidth scaling")
    return values


def _validate_grid(bandwidths: Sequence[float], horizon: int) -> Vector:
    widths = tuple(bandwidths)
    if not widths or any(not math.isfinite(width) or width <= 0 for width in widths):
        raise ValueError("trajectory baseline bandwidths must be positive and finite")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("trajectory baseline horizon must be a positive integer")
    return widths


def _training_data(
    bank: Iterable[Rollout], bandwidths: Sequence[float], horizon: int
) -> tuple[Vector, list[tuple[Vector, bool]]]:
    widths = _validate_grid(bandwidths, horizon)
    records = []
    for rollout in bank:
        if len(rollout.features) != horizon:
            continue
        failed = rollout.first_event is not None
        if failed or rollout.observed_steps == horizon:
            records.append((_flatten(rollout.features, widths, horizon), failed))
    return widths, records


def _log_mean_exp(values: Sequence[float]) -> float:
    largest = max(values)
    if not math.isfinite(largest):
        return largest
    mean = math.fsum(math.exp(value - largest) for value in values) / len(values)
    return largest + math.log(mean)


@dataclass(frozen=True)
class TrajectoryLogKDE:
    """Class-conditional Gaussian log-density contrast over full trajectories."""

    bandwidths: Vector
    horizon: int
    failures: tuple[Vector, ...]
    successes: tuple[Vector, ...]

    @classmethod
    def fit(
        cls, bank: Iterable[Rollout], bandwidths: Sequence[float], horizon: int
    ) -> TrajectoryLogKDE | None:
        widths, records = _training_data(bank, bandwidths, horizon)
        failures = tuple(features for features, failed in records if failed)
        successes = tuple(features for features, failed in records if not failed)
        if not failures or not successes:
            return None
        return cls(widths, horizon, failures, successes)

    @property
    def training_count(self) -> int:
        return len(self.failures) + len(self.successes)

    @property
    def prior_logit(self) -> float:
        return math.log(len(self.failures)) - math.log(len(self.successes))

    def predict(self, features: FeatureSequence) -> Risk:
        query = _flatten(features, self.bandwidths, self.horizon)

        def log_affinity(reference: Vector) -> float:
            # Remove the common -||query||^2/2 term from the contrast. Otherwise
            # a distant query can make both densities underflow to zero.
            return math.fsum(
                (value - 0.5 * center) * center for value, center in zip(query, reference)
            )

        failure = _log_mean_exp(tuple(log_affinity(row) for row in self.failures))
        success = _log_mean_exp(tuple(log_affinity(row) for row in self.successes))
        return Risk.from_logit(self.prior_logit + (failure - success))


class RawKernelTrajectory(TrajectoryLogKDE):
    """Uncalibrated raw Gaussian class-affinity contrast over full trajectories."""

    def score(self, features: FeatureSequence) -> float:
        """Return mean failure affinity minus mean success affinity."""
        query = _flatten(features, self.bandwidths, self.horizon)

        def affinity(reference: Vector) -> float:
            distance = math.dist(query, reference)
            return math.exp(-0.5 * distance * distance)

        failure = math.fsum(affinity(row) for row in self.failures) / len(self.failures)
        success = math.fsum(affinity(row) for row in self.successes) / len(self.successes)
        return failure - success

    def predict(self, features: FeatureSequence) -> Risk:
        """Map the bounded raw contrast to a score around the training class prior."""
        return Risk.from_logit(self.prior_logit + self.score(features))


@dataclass(frozen=True)
class TrajectoryLogistic:
    """Regularized logistic regression with train-only feature normalization."""

    bandwidths: Vector
    horizon: int
    center: Vector
    scale: Vector
    weights: Vector
    intercept: float
    training_count: int

    @classmethod
    def fit(
        cls,
        bank: Iterable[Rollout],
        bandwidths: Sequence[float],
        horizon: int,
        *,
        l2: float = 0.01,
        iterations: int = 200,
    ) -> TrajectoryLogistic | None:
        widths, examples = _training_data(bank, bandwidths, horizon)
        return cls._fit_rows(widths, horizon, examples, l2=l2, iterations=iterations)

    @classmethod
    def _fit_rows(
        cls,
        widths: Vector,
        horizon: int,
        examples: list[tuple[Vector, bool]],
        *,
        l2: float,
        iterations: int,
    ) -> TrajectoryLogistic | None:
        if not math.isfinite(l2) or l2 <= 0 or type(iterations) is not int or iterations < 1:
            raise ValueError("trajectory logistic regression needs positive regularization and iterations")
        count = len(examples)
        failed_count = sum(failed for _, failed in examples)
        if not 0 < failed_count < count:
            return None

        columns = tuple(zip(*(features for features, _ in examples)))
        center = tuple(math.fsum(column) / count for column in columns)
        scale = []
        for mean, column in zip(center, columns):
            deviation = math.hypot(*(value - mean for value in column)) / math.sqrt(count)
            scale.append(deviation if deviation > 0 else 1.0)
        standardized = [
            (tuple((value - mean) / spread for value, mean, spread in zip(features, center, scale)), failed)
            for features, failed in examples
        ]

        weights = [0.0] * len(center)
        intercept = math.log(failed_count) - math.log(count - failed_count)
        # An upper bound on logistic-loss curvature gives a stable fixed step.
        mean_squared_norm = math.fsum(
            value * value for features, _ in standardized for value in features
        ) / count
        step = 1.0 / (0.25 * (1 + mean_squared_norm) + l2)
        inverse_count = 1.0 / count
        for _ in range(iterations):
            weight_gradients = [l2 * weight for weight in weights]
            intercept_gradient = 0.0
            for features, failed in standardized:
                logit = intercept + math.fsum(weight * value for weight, value in zip(weights, features))
                score = Risk.from_logit(logit)
                residual = (-math.exp(score.log_survival) if failed else score.failure_probability) * inverse_count
                intercept_gradient += residual
                for index, value in enumerate(features):
                    weight_gradients[index] += residual * value
            if max(abs(intercept_gradient), max(map(abs, weight_gradients))) < 1e-7:
                break
            intercept -= step * intercept_gradient
            weights = [
                weight - step * gradient for weight, gradient in zip(weights, weight_gradients)
            ]
        return cls(widths, horizon, center, tuple(scale), tuple(weights), intercept, count)

    def predict(self, features: FeatureSequence) -> Risk:
        query = _flatten(features, self.bandwidths, self.horizon)
        logit = self.intercept + math.fsum(
            weight * ((value - mean) / spread)
            for weight, value, mean, spread in zip(self.weights, query, self.center, self.scale)
        )
        return Risk.from_logit(logit)


@dataclass(frozen=True)
class StepwiseLogistic:
    """Shared logistic head trained on the bank's observed at-risk transitions."""

    step_model: TrajectoryLogistic
    horizon: int

    @classmethod
    def fit(
        cls,
        bank: Iterable[Rollout],
        bandwidths: Sequence[float],
        horizon: int,
        *,
        l2: float = 0.01,
        iterations: int = 200,
    ) -> StepwiseLogistic | None:
        widths = _validate_grid(bandwidths, horizon)
        examples = [
            (_flatten((features,), widths, 1), event)
            for rollout in bank for features, event in rollout.at_risk()
        ]
        model = TrajectoryLogistic._fit_rows(widths, 1, examples, l2=l2, iterations=iterations)
        return cls(model, horizon) if model is not None else None

    @property
    def training_count(self) -> int:
        return self.step_model.training_count

    def predict(self, features: FeatureSequence) -> Risk:
        if len(features) != self.horizon:
            raise ValueError("stepwise baseline requires the full evaluation horizon")
        scores = tuple(self.step_model.predict((row,)) for row in features)
        log_survival = math.fsum(score.log_survival for score in scores)
        failure_probability = -math.expm1(log_survival)
        if failure_probability:
            return Risk(failure_probability, log_survival, math.log(failure_probability))
        # Preserve the log score even if every step's failure probability underflows.
        prefix_log_survival = 0.0
        first_event_log_masses = []
        for score in scores:
            first_event_log_masses.append(prefix_log_survival + score.log_failure)
            prefix_log_survival += score.log_survival
        log_failure = _log_mean_exp(first_event_log_masses) + math.log(self.horizon)
        return Risk(failure_probability, log_survival, log_failure)


class EventOnlyModel:
    """Run Eq. (3) on a declared subset such as time, position and action."""

    def __init__(
        self,
        bank: Iterable[Rollout],
        bandwidths: Sequence[float],
        feature_indices: Sequence[int],
        **hazard_options,
    ) -> None:
        indices = tuple(feature_indices)
        if not indices or any(type(index) is not int or not 0 <= index < len(bandwidths) for index in indices):
            raise ValueError("event-only feature indices must be within the bank feature dimension")
        if len(set(indices)) != len(indices):
            raise ValueError("event-only feature indices must be unique")
        self.feature_indices = indices
        self.dimension = len(bandwidths)
        reduced = (
            Rollout(row.group, self._project(row.features), row.observed_steps, row.first_event)
            for row in bank
        )
        self.model = HazardModel(
            reduced, tuple(bandwidths[index] for index in indices), **hazard_options
        )

    def _project(self, features: FeatureSequence) -> tuple[Vector, ...]:
        if any(len(row) != self.dimension for row in features):
            raise ValueError("event-only vectors must have the bank feature dimension")
        return tuple(tuple(row[index] for index in self.feature_indices) for row in features)

    def predict(
        self, features: FeatureSequence, *, exclude_groups: Iterable[str] = ()
    ) -> Prediction:
        return self.model.predict(self._project(features), exclude_groups=exclude_groups)
