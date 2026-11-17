from __future__ import annotations

import pytest
import torch

import openpinn as op


def _two_field_problem(residual: object | None = None) -> op.PDEProblem:
    def valid_residual(
        coordinates: torch.Tensor, prediction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del coordinates
        return {
            "field_u": prediction[:, 0:1],
            "field_v": prediction[:, 1:2],
        }

    return op.PDEProblem(
        op.Interval(-1.0, 1.0),
        residual=valid_residual if residual is None else residual,  # type: ignore[arg-type]
        field_names=("u", "v"),
        output_dim=2,
        name="two_field_test",
    )


def test_problem_defaults_preserve_single_field_compatibility() -> None:
    problem = op.make_poisson_problem()

    assert problem.field_names == ("u",)
    assert problem.output_dim == 1


def test_problem_declares_and_validates_two_field_residuals() -> None:
    problem = _two_field_problem()
    model = op.PINNModel(op.MLP(1, 2, hidden_layers=(4,)))
    coordinates = torch.tensor([[-0.5], [0.5]], requires_grad=True)
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    residuals = problem.evaluate_residual(coordinates, prediction)

    assert problem.field_names == ("u", "v")
    assert problem.output_dim == 2
    assert model.output_dim == problem.output_dim
    assert set(residuals) == {"field_u", "field_v"}
    assert all(value.shape == (2, 1) for value in residuals.values())


@pytest.mark.parametrize(
    ("residual", "message"),
    (
        (lambda coordinates, prediction: prediction[:, 0], "shape"),
        (lambda coordinates, prediction: prediction[:1, 0:1], "batch size"),
    ),
)
def test_problem_rejects_invalid_residual_shapes(
    residual: object, message: str
) -> None:
    problem = _two_field_problem(residual)
    coordinates = torch.tensor([[-0.5], [0.5]], requires_grad=True)
    prediction = torch.ones((2, 2), requires_grad=True)

    with pytest.raises(ValueError, match=message):
        problem.evaluate_residual(coordinates, prediction)


def test_problem_rejects_prediction_width_and_field_name_mismatch() -> None:
    problem = _two_field_problem()
    coordinates = torch.tensor([[-0.5], [0.5]], requires_grad=True)

    with pytest.raises(ValueError, match="prediction has width"):
        problem.evaluate_residual(coordinates, torch.ones((2, 1)))
    with pytest.raises(ValueError, match="field_names has length"):
        op.PDEProblem(
            op.Interval(-1.0, 1.0),
            residual=lambda x, y: y[:, :1],
            field_names=("u", "v"),
            output_dim=1,
        )
    with pytest.raises(TypeError, match="positive integer"):
        op.PDEProblem(
            op.Interval(-1.0, 1.0),
            residual=lambda x, y: y[:, :1],
            output_dim=1.5,  # type: ignore[arg-type]
        )


def test_loss_composer_rejects_weight_names_not_produced_by_formulation() -> None:
    composer = op.LossComposer({"pde_typo": 2.0})

    with pytest.raises(ValueError, match="do not match"):
        composer({"pde": torch.tensor(1.0)})


def test_adaptive_loss_composer_ascends_positive_term_weights() -> None:
    composer = op.AdaptiveLossComposer(
        {"pde": 1.0, "boundary": 1.0}, lr=0.1, max_weight=10.0
    )
    initial = {
        name: float(value)
        for name, value in composer.current_weights().items()
    }
    terms = {"pde": torch.tensor(4.0), "boundary": torch.tensor(1.0)}

    total = composer(terms)
    updated = composer.adapt(terms)

    assert float(total) == pytest.approx(5.0)
    assert updated["pde"] > initial["pde"]
    assert updated["boundary"] > initial["boundary"]
    assert updated["pde"] > updated["boundary"]
    assert all(value > 0.0 for value in updated.values())
