# PINN 2D Poisson M19-alt REPRO

## Problem
- Domain: Polygon [-1,1]^2
- PDE: -u_xx - u_yy - 2*pi^2*sin(pi*x)*sin(pi*y) = 0
- BC: Dirichlet soft zero (DirichletBC(0))
- Exact: u = sin(pi*x)*sin(pi*y)
- Note: hard_boundary not used (library supports it only for built-in poisson/burgers)

## Stack
- modified_mlp / gated_mlp
- fourier / random_fourier (fallback without if incompatible)
- adaptive_weights / relobralo
- adaptive_sampling / residual_adaptive
- optional adam_lbfgs polish

## Run
The maintainer trained this case outside the repository. The training script is
not part of the product tree. Recorded settings: seed=0, device=cuda, lr=0.001,
openpinn 0.2.0. This notebook does not retrain that run.

## Result
- stop_steps: 5000
- relative_l2_error: 0.033738549798727036
- passed (rel_L2 < 0.05): True
- passed_soft (rel_L2 < 0.1): True
- elapsed_sec: 251.681
- seed: 0, device: cuda, lr: 0.001
- openpinn: 0.2.0
