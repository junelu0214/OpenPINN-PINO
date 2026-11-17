from __future__ import annotations

import torch

import openpinn as op
from openpinn.experiment import _model


def test_poisson_hard_constraint_does_not_embed_the_reference_solution() -> None:
    problem = op.make_poisson_problem()
    model = _model(
        {
            "model": {
                "hidden_layers": [8],
                "activation": "tanh",
                "constraint": {"name": "hard"},
            }
        },
        problem,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    interior = torch.tensor([[0.5]])

    prediction = model(interior)
    reference = problem.exact(interior)

    torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=1e-6, rtol=0.0)
    assert not torch.allclose(prediction, reference)
