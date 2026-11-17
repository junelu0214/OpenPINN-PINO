"""Uniform closure adapters for PyTorch optimizers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch
from torch import Tensor, nn

LossClosure = Callable[[], Tensor]


class OptimizerAdapter:
    """Wrap a PyTorch optimizer behind a closure-based PINN step."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.closure_calls = 0

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Expose underlying optimizer parameter groups."""

        return self.optimizer.param_groups

    def _backward_closure(self, loss_closure: LossClosure) -> Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if loss.ndim != 0:
            raise ValueError("Optimizer closure must return a scalar loss.")
        loss.backward()
        self.closure_calls += 1
        return loss

    def step(self, loss_closure: LossClosure) -> Tensor:
        """Perform one optimizer step and return its evaluated loss."""

        loss = self._backward_closure(loss_closure)
        self.optimizer.step()
        return loss.detach()

    def zero_grad(self) -> None:
        """Clear gradients."""

        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state plus adapter accounting."""

        return {
            "optimizer": self.optimizer.state_dict(),
            "closure_calls": self.closure_calls,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the underlying optimizer state."""

        values = dict(state)
        if "optimizer" in values:
            self.optimizer.load_state_dict(dict(values["optimizer"]))
            self.closure_calls = int(values.get("closure_calls", 0))
        else:  # Backward compatibility with early starter checkpoints.
            self.optimizer.load_state_dict(values)


class AdamAdapter(OptimizerAdapter):
    """Adapter for :class:`torch.optim.Adam`."""

    def __init__(self, parameters: Iterable[nn.Parameter], lr: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(torch.optim.Adam(parameters, lr=lr, **kwargs))


class AdamWAdapter(OptimizerAdapter):
    """Adapter for :class:`torch.optim.AdamW`."""

    def __init__(self, parameters: Iterable[nn.Parameter], lr: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(torch.optim.AdamW(parameters, lr=lr, **kwargs))


class LBFGSAdapter(OptimizerAdapter):
    """Full-batch L-BFGS adapter using PyTorch's repeated closure contract."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        lr: float = 1.0,
        *,
        max_iter: int = 20,
        history_size: int = 100,
        line_search_fn: str | None = "strong_wolfe",
        **kwargs: Any,
    ) -> None:
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn=line_search_fn,
            **kwargs,
        )
        super().__init__(optimizer)

    def step(self, loss_closure: LossClosure) -> Tensor:
        """Perform one outer L-BFGS step with a deterministic full batch."""

        result = self.optimizer.step(lambda: self._backward_closure(loss_closure))
        if isinstance(result, Tensor):
            return result.detach()
        return torch.as_tensor(result)


class StagedOptimizerAdapter(OptimizerAdapter):
    """Run explicit optimizer stages on one shared parameter set."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        stages: Iterable[Mapping[str, Any]],
    ) -> None:
        parameter_list = list(parameters)
        if not parameter_list:
            raise ValueError("StagedOptimizerAdapter requires trainable parameters.")
        self.adapters: list[OptimizerAdapter] = []
        self.stage_plan: list[dict[str, Any]] = []
        for stage in stages:
            values = dict(stage)
            name = str(values.pop("name", values.pop("kind", "")))
            steps = int(values.pop("steps", 0))
            if not name or steps <= 0:
                raise ValueError("Each optimizer stage requires a name and positive steps.")
            normalized = name.lower().replace("-", "").replace("_", "")
            if normalized in {"adamlbfgs", "staged"}:
                raise ValueError("Staged optimizers cannot contain another staged optimizer.")
            learning_rate = values.pop("lr", values.pop("learning_rate", None))
            adapter = make_optimizer(
                name,
                parameter_list,
                lr=learning_rate,
                **values,
            )
            self.adapters.append(adapter)
            self.stage_plan.append({"name": normalized, "steps": steps})
        if not self.adapters:
            raise ValueError("At least one optimizer stage is required.")
        self.current_stage_index = 0
        self.current_stage_step = 0
        self.total_outer_steps = 0
        self.last_stage_index = 0

    @property
    def closure_calls(self) -> int:
        """Return closure evaluations accumulated across all stages."""

        return sum(adapter.closure_calls for adapter in self.adapters)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Expose the active stage's parameter groups."""

        return self.adapters[self.current_stage_index].param_groups

    @property
    def planned_outer_steps(self) -> int:
        """Return the configured sum of stage outer steps."""

        return sum(int(stage["steps"]) for stage in self.stage_plan)

    def step(self, loss_closure: LossClosure) -> Tensor:
        """Take one step with the current stage, then advance if complete."""

        index = self.current_stage_index
        self.last_stage_index = index
        result = self.adapters[index].step(loss_closure)
        self.current_stage_step += 1
        self.total_outer_steps += 1
        planned = int(self.stage_plan[index]["steps"])
        if self.current_stage_step >= planned and index < len(self.adapters) - 1:
            self.current_stage_index += 1
            self.current_stage_step = 0
        return result

    def zero_grad(self) -> None:
        """Clear gradients on the active optimizer."""

        self.adapters[self.current_stage_index].zero_grad()

    def diagnostics(self) -> dict[str, Any]:
        """Return stage names, planned steps and actual closure calls."""

        return {
            "planned_outer_steps": self.planned_outer_steps,
            "completed_outer_steps": self.total_outer_steps,
            "stages": [
                {
                    **stage,
                    "closure_calls": adapter.closure_calls,
                }
                for stage, adapter in zip(self.stage_plan, self.adapters, strict=True)
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        """Return every stage optimizer plus switching counters."""

        return {
            "stage_plan": [dict(stage) for stage in self.stage_plan],
            "adapters": [adapter.state_dict() for adapter in self.adapters],
            "current_stage_index": self.current_stage_index,
            "current_stage_step": self.current_stage_step,
            "total_outer_steps": self.total_outer_steps,
            "last_stage_index": self.last_stage_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a configuration-compatible staged optimizer."""

        restored_plan = [dict(stage) for stage in state["stage_plan"]]
        if restored_plan != self.stage_plan:
            raise ValueError("Optimizer stage plan differs from checkpoint.")
        restored_adapters = list(state["adapters"])
        if len(restored_adapters) != len(self.adapters):
            raise ValueError("Optimizer stage count differs from checkpoint.")
        for adapter, adapter_state in zip(
            self.adapters, restored_adapters, strict=True
        ):
            adapter.load_state_dict(adapter_state)
        self.current_stage_index = int(state["current_stage_index"])
        self.current_stage_step = int(state["current_stage_step"])
        self.total_outer_steps = int(state["total_outer_steps"])
        self.last_stage_index = int(state.get("last_stage_index", 0))


Adam = AdamAdapter
AdamW = AdamWAdapter
LBFGS = LBFGSAdapter


def make_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    *,
    lr: float | None = None,
    **kwargs: Any,
) -> OptimizerAdapter:
    """Construct a supported optimizer adapter by name."""

    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized == "adam":
        return AdamAdapter(parameters, lr=1e-3 if lr is None else lr, **kwargs)
    if normalized == "adamw":
        return AdamWAdapter(parameters, lr=1e-3 if lr is None else lr, **kwargs)
    if normalized in {"lbfgs", "lbgfs"}:
        return LBFGSAdapter(parameters, lr=1.0 if lr is None else lr, **kwargs)
    if normalized in {"adamlbfgs", "staged"}:
        if lr is not None:
            raise ValueError("Set learning rates inside optimizer.stages.")
        stages = kwargs.pop("stages", None)
        if stages is None or kwargs:
            raise ValueError("A staged optimizer requires only an explicit stages list.")
        return StagedOptimizerAdapter(parameters, stages)
    raise ValueError(f"Unsupported optimizer {name!r}; choose Adam, AdamW or LBFGS.")
