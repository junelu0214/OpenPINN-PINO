"""Explicit full-batch PINN training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .formulations import FormulationProtocol, StrongFormulation
from .losses import (
    AdaptiveLossComposer,
    LossComposer,
    MagnitudePowerLossComposer,
    PointwiseAdaptiveLossComposer,
    ReLoBRaLoLossComposer,
    UncertaintyLossComposer,
)
from .optimizers import OptimizerAdapter, make_optimizer
from .problems import PDEProblem
from .sampling import SampleBatch, SamplerProtocol


class TrainingHistory(Sequence):
    """List-like training records with convenient column access."""

    def __init__(self, records: Sequence[Mapping[str, float]] | None = None) -> None:
        self.records: list[dict[str, float]] = [dict(record) for record in (records or ())]

    def append(self, record: Mapping[str, float]) -> None:
        """Append a numeric record."""

        self.records.append({str(key): float(value) for key, value in record.items()})

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter(self.records)

    def __getitem__(self, item: int | slice | str) -> Any:
        if isinstance(item, str):
            return [record[item] for record in self.records if item in record]
        return self.records[item]

    @property
    def last(self) -> dict[str, float]:
        """Return the final record."""

        if not self.records:
            raise IndexError("Training history is empty.")
        return self.records[-1]

    def as_list(self) -> list[dict[str, float]]:
        """Return a copy suitable for serialization."""

        return [dict(record) for record in self.records]


class Trainer:
    """Train a model against a formulation with a fixed public interface.

    The starter uses full-batch collocation. Adaptive behavior remains
    explicit through separately configured components.
    """

    def __init__(
        self,
        model: nn.Module,
        formulation: FormulationProtocol | PDEProblem,
        sampler: SamplerProtocol | None = None,
        *,
        optimizer: str | OptimizerAdapter = "adam",
        lr: float | None = 1e-3,
        loss_composer: (
            LossComposer
            | AdaptiveLossComposer
            | MagnitudePowerLossComposer
            | PointwiseAdaptiveLossComposer
            | UncertaintyLossComposer
            | ReLoBRaLoLossComposer
            | None
        ) = None,
        loss_weights: Mapping[str, float] | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.formulation = (
            StrongFormulation(formulation)
            if isinstance(formulation, PDEProblem)
            else formulation
        )
        if not isinstance(self.formulation, FormulationProtocol):
            raise TypeError(
                "formulation must be a PDEProblem or implement FormulationProtocol."
            )
        self.problem = self.formulation.problem
        if sampler is not None and not isinstance(sampler, SamplerProtocol):
            raise TypeError("sampler must implement SamplerProtocol.")
        self.sampler = sampler

        try:
            reference_parameter = next(self.model.parameters())
            inferred_device = reference_parameter.device
            inferred_dtype = reference_parameter.dtype
        except StopIteration:
            inferred_device = torch.device("cpu")
            inferred_dtype = torch.get_default_dtype()
        self.device = torch.device(device) if device is not None else inferred_device
        self.dtype = dtype if dtype is not None else inferred_dtype
        self.model.to(device=self.device, dtype=self.dtype)

        if loss_composer is not None and loss_weights is not None:
            raise ValueError("Pass loss_composer or loss_weights, not both.")
        self.loss_composer = loss_composer or LossComposer(loss_weights)
        if isinstance(self.loss_composer, nn.Module):
            self.loss_composer.to(device=self.device, dtype=self.dtype)
        bind_model = getattr(self.loss_composer, "bind_model", None)
        if callable(bind_model):
            bind_model(self.model)
        if isinstance(optimizer, OptimizerAdapter):
            self.optimizer = optimizer
        else:
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if not trainable:
                raise ValueError("The model has no trainable parameters.")
            self.optimizer = make_optimizer(
                optimizer,
                trainable,
                lr=lr,
                **dict(optimizer_kwargs or {}),
            )
        self.history = TrainingHistory()
        self.steps_completed = 0
        self.elapsed_seconds = 0.0
        self._active_batch: SampleBatch | None = None
        self._resample_every: int | None = None

    def _loss_terms(self, batch: SampleBatch) -> dict[str, Tensor]:
        return dict(self.formulation.loss_terms(self.model, batch))

    def _loss_payload(
        self, batch: SampleBatch
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor] | None]:
        bind_batch = getattr(self.loss_composer, "bind_batch", None)
        if callable(bind_batch):
            bind_batch(batch, self.problem)
        if getattr(self.loss_composer, "requires_pointwise_terms", False):
            pointwise_method = getattr(self.formulation, "pointwise_loss_terms", None)
            if not callable(pointwise_method):
                raise TypeError(
                    "The selected pointwise loss method requires a formulation "
                    "with pointwise_loss_terms(model, batch)."
                )
            pointwise = dict(pointwise_method(self.model, batch))
            scalar = {name: values.mean() for name, values in pointwise.items()}
            compose_pointwise = getattr(self.loss_composer, "compose_pointwise")
            return compose_pointwise(pointwise), scalar, pointwise
        terms = self._loss_terms(batch)
        if callable(self.loss_composer):
            total = self.loss_composer(terms)
        else:
            compose = getattr(self.loss_composer, "compose", None)
            if not callable(compose):
                raise TypeError(
                    "Loss composer must be callable or expose compose(terms)."
                )
            total = compose(terms)
        return total, terms, None

    def loss(self, batch: SampleBatch) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate total and named losses for a batch."""

        total, terms, _ = self._loss_payload(batch)
        return total, terms

    def fit(
        self,
        steps: int | None = None,
        *,
        epochs: int | None = None,
        sampler: SamplerProtocol | None = None,
        resample_every: int = 1,
        log_every: int = 0,
        callback: Callable[[dict[str, float]], None] | None = None,
    ) -> TrainingHistory:
        """Run full-batch optimization and return cumulative history."""

        if steps is None:
            steps = 1000 if epochs is None else epochs
        elif epochs is not None and int(epochs) != int(steps):
            raise ValueError("steps and epochs disagree; pass only one.")
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if resample_every <= 0:
            raise ValueError("resample_every must be positive.")
        active_sampler = sampler or self.sampler
        if active_sampler is None:
            raise ValueError("Trainer.fit requires a sampler.")
        if not isinstance(active_sampler, SamplerProtocol):
            raise TypeError("sampler must implement SamplerProtocol.")

        if self._resample_every is not None and self._resample_every != resample_every:
            self._active_batch = None
        self._resample_every = resample_every
        started = time.perf_counter()
        elapsed_before_session = self.elapsed_seconds
        for local_step in range(1, steps + 1):
            if self._active_batch is None or self.steps_completed % resample_every == 0:
                update_sampler = getattr(active_sampler, "update", None)
                if self.steps_completed > 0 and callable(update_sampler):
                    update_sampler(
                        self.model,
                        device=self.device,
                        dtype=self.dtype,
                    )
                self._active_batch = active_sampler.sample(
                    self.problem, device=self.device, dtype=self.dtype, requires_grad=True
                )
            batch = self._active_batch
            latest_terms: dict[str, Tensor] = {}
            latest_pointwise_terms: dict[str, Tensor] | None = None

            def closure() -> Tensor:
                nonlocal latest_terms, latest_pointwise_terms
                total, terms, pointwise = self._loss_payload(batch)  # type: ignore[arg-type]
                latest_terms = terms
                latest_pointwise_terms = pointwise
                return total

            returned_loss = self.optimizer.step(closure)
            adaptive_weights: dict[str, float] = {}
            adapt = getattr(self.loss_composer, "adapt", None)
            if callable(adapt):
                adaptive_weights = adapt(
                    latest_pointwise_terms
                    if latest_pointwise_terms is not None
                    else latest_terms
                )
            self.steps_completed += 1
            record: dict[str, float] = {
                "step": float(self.steps_completed),
                "elapsed_seconds": float(
                    elapsed_before_session + time.perf_counter() - started
                ),
            }
            for name, value in latest_terms.items():
                record[name] = float(value.detach().cpu())
            if latest_terms:
                weighted_terms = getattr(
                    self.loss_composer, "weighted_terms", None
                )
                if callable(weighted_terms):
                    with torch.no_grad():
                        weighted = weighted_terms(latest_terms)
                        total = torch.stack(tuple(weighted.values())).sum()
                        if getattr(self.loss_composer, "reduction", "sum") == "mean":
                            total = total / len(weighted)
                else:
                    total = returned_loss
                record["total"] = float(total.detach().cpu())
            else:
                record["total"] = float(returned_loss.detach().cpu())
            for name, value in adaptive_weights.items():
                record[f"weight_{name}"] = value
            stage_index = getattr(self.optimizer, "last_stage_index", None)
            if stage_index is not None:
                record["optimizer_stage"] = float(stage_index)
            self.history.append(record)
            if callback is not None:
                callback(dict(record))
            if log_every and (local_step == 1 or local_step % log_every == 0 or local_step == steps):
                print(f"step={self.steps_completed} total={record['total']:.6e}")
        self.elapsed_seconds = float(elapsed_before_session + time.perf_counter() - started)
        return self.history

    train = fit

    def evaluate_loss(
        self,
        batch: SampleBatch | None = None,
        *,
        sampler: SamplerProtocol | None = None,
    ) -> dict[str, float]:
        """Evaluate named and total losses on a fresh or supplied batch."""

        if batch is None:
            active_sampler = sampler or self.sampler
            if active_sampler is None:
                raise ValueError("A sampler or batch is required.")
            batch = active_sampler.sample(
                self.problem, device=self.device, dtype=self.dtype, requires_grad=True
            )
        total, terms = self.loss(batch)
        values = {name: float(value.detach().cpu()) for name, value in terms.items()}
        values["total"] = float(total.detach().cpu())
        return values

    def predict(self, coordinates: Tensor) -> Tensor:
        """Evaluate model predictions without constructing a derivative graph."""

        coordinates = coordinates.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        with torch.no_grad():
            result = self.model(coordinates)
        self.model.train()
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, counters and reproducibility state."""

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_composer": (
                self.loss_composer.training_state_dict()
                if callable(getattr(self.loss_composer, "training_state_dict", None))
                else None
            ),
            "steps_completed": self.steps_completed,
            "history": self.history.as_list(),
            "elapsed_seconds": self.elapsed_seconds,
            "sampler": self.sampler.state_dict() if self.sampler is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "active_batch": (
                {
                    name: value.detach().cpu()
                    for name, value in self._active_batch.as_dict().items()
                }
                if self._active_batch is not None
                else None
            ),
            "resample_every": self._resample_every,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore model, optimizer and training counters."""

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        composer_state = state.get("loss_composer")
        if composer_state is not None:
            load_composer = getattr(self.loss_composer, "load_training_state_dict", None)
            if not callable(load_composer):
                raise ValueError(
                    "Checkpoint contains adaptive loss state but Trainer has a fixed composer."
                )
            load_composer(composer_state)
        self.steps_completed = int(state.get("steps_completed", 0))
        self.history = TrainingHistory(state.get("history", ()))
        self.elapsed_seconds = float(
            state.get(
                "elapsed_seconds",
                max(self.history["elapsed_seconds"], default=0.0),
            )
        )
        sampler_state = state.get("sampler")
        if sampler_state is not None:
            if self.sampler is None:
                raise ValueError("Checkpoint contains sampler state but Trainer has no sampler.")
            self.sampler.load_state_dict(sampler_state)
        torch_rng_state = state.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_state = state.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        active_batch = state.get("active_batch")
        if active_batch is None:
            self._active_batch = None
        else:
            self._active_batch = SampleBatch(
                interior=active_batch["interior"],
                boundary=active_batch["boundary"],
                initial=active_batch["initial"],
                interior_weights=active_batch.get("interior_weights"),
            ).to(device=self.device, dtype=self.dtype)
            self._active_batch.interior.requires_grad_(True)
            self._active_batch.boundary.requires_grad_(True)
            self._active_batch.initial.requires_grad_(True)
        stored_resample_every = state.get("resample_every")
        self._resample_every = (
            int(stored_resample_every) if stored_resample_every is not None else None
        )

    def save(self, path: str | Path) -> Path:
        """Write a restorable training checkpoint."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)
        return target

    def load(self, path: str | Path, *, map_location: str | torch.device | None = None) -> None:
        """Load a checkpoint written by :meth:`save`."""

        state = torch.load(path, map_location=map_location or self.device, weights_only=True)
        self.load_state_dict(state)
