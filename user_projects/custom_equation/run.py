"""The shortest custom-PDE workflow: define once, select a method by name."""

import openpinn as op

from user_projects.custom_equation.problem import build_problem


result = op.solve(
    build_problem(),
    method="adaptive_sampling",
    variant="residual_adaptive",
    steps=200,
)
print(result.metrics)
