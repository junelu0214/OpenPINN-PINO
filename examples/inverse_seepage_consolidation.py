"""Run PINN seepage inversion and DeepONet consolidation-family inversion."""

from __future__ import annotations

import torch

import openpinn as op


# Route 1: a PINN learns one steady seepage field and its permeability.
seepage_x = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).reshape(-1, 1)
seepage_p = op.darcy_pressure(seepage_x, permeability=2.0, source=1.2)
pinn = op.inverse_darcy_pinn(
    seepage_x, seepage_p, source=1.2,
    initial_permeability=0.7, permeability_bounds=(0.2, 5.0),
)
print("PINN seepage permeability:", pinn.estimated_permeability)

# Route 2: DeepONet learns a transient (cv, U0) consolidation family. Its
# weights are frozen before identifying cv for a previously unseen member.
family = op.train_consolidation_family_operator(
    coefficient_bounds=(0.05, 0.8), amplitude_bounds=(0.5, 2.0),
)
consolidation_zt = torch.tensor([
    [0.2, 0.05], [0.2, 0.2], [0.2, 0.5],
    [0.5, 0.05], [0.5, 0.2], [0.5, 0.5],
    [0.8, 0.05], [0.8, 0.2], [0.8, 0.5],
])
consolidation_u = op.consolidation_pressure(
    consolidation_zt, consolidation_coefficient=0.3, amplitude=1.2,
)
operator_inverse = op.inverse_consolidation_operator(
    family, consolidation_zt, consolidation_u,
    amplitude=1.2, initial_coefficient=0.2,
)
print("DeepONet consolidation coefficient:", operator_inverse.estimated_coefficient)
