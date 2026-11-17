"""Minimal Poisson example using the public OpenPINN API."""

from __future__ import annotations

import argparse
import json
import math

import torch

import openpinn as op


def run(*, steps: int = 10, seed: int = 0) -> tuple[op.Trainer, op.TrainingHistory]:
    """Build and train Poisson through the public component interfaces."""

    if steps <= 0:
        raise ValueError("steps must be positive.")

    torch.manual_seed(seed)

    problem = op.make_poisson_problem()
    network = op.MLP(1, 1, hidden_layers=(64, 64, 64))
    model = op.PINNModel(network)
    points = op.UniformSampler(
        problem,
        n_interior=128,
        n_boundary=32,
        seed=seed,
    )

    trainer = op.Trainer(
        model,
        problem,
        points,
        optimizer="adam",
        lr=1e-3,
    )
    history = trainer.fit(steps=steps)
    return trainer, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a short Poisson example through the public OpenPINN API."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Training steps; use 5 for a quick check and a larger value for a full run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, history = run(steps=args.steps, seed=args.seed)
    final_loss = float(history.last["total"])
    print(
        json.dumps(
            {
                "example": "poisson_api_quickstart",
                "status": "completed_with_finite_outputs"
                if math.isfinite(final_loss)
                else "failed_non_finite_output",
                "steps": len(history),
                "seed": args.seed,
                "final_total_loss": final_loss,
                "accuracy_claim": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
