# Computational and verification plan

Use a small PyTorch coordinate network and OpenPINN automatic differentiation.
Minimize the ODE residual and initial-condition loss for the smoke budget from
`OPENPINN_BUDGET_JSON`. Evaluate on independent points against the analytical
reference, record relative L2 error, and do not mark accuracy as verified because
this example declares no acceptance threshold.
