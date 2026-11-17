"""Agent-authored example for a problem absent from the built-in registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from openpinn import grad


torch.manual_seed(7)
budget = json.loads(os.environ.get("OPENPINN_BUDGET_JSON", "{}"))
steps = int(budget.get("steps", 40))
output_dir = Path(os.environ["OPENPINN_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
history: list[float] = []
for _ in range(steps):
    coordinates = torch.linspace(0.0, 1.0, 33).reshape(-1, 1).requires_grad_(True)
    prediction = model(coordinates)
    residual = grad(prediction, coordinates) + prediction
    initial = model(torch.zeros((1, 1))) - 1.0
    loss = residual.square().mean() + 10.0 * initial.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(float(loss.detach()))

evaluation = torch.linspace(0.0, 1.0, 101).reshape(-1, 1)
with torch.no_grad():
    prediction = model(evaluation)
    reference = torch.exp(-evaluation)
    relative_l2 = float(torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference))

(output_dir / "loss_history.json").write_text(
    json.dumps(history, indent=2), encoding="utf-8"
)
(output_dir / "run_result.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "budget_exhausted",
            "summary": "Agent-authored bounded ODE PINN run completed.",
            "metrics": {
                "final_loss": history[-1],
                "relative_l2": relative_l2,
                "steps": steps,
            },
            "claims": {
                "finite_outputs": True,
                "convergence_verified": False,
                "accuracy_verified": False,
            },
            "artifacts": ["loss_history.json"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
