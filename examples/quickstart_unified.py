from pathlib import Path

import openpinn as op

# 保存为项目 examples/ 目录中的 .py 文件后，可直接点击 VS Code 三角形运行。
PROBLEM = "tunnel_model1"

STEPS = 10000
LOG_EVERY = 100
SEED = 0

METHODS = [
    op.method(
        "adaptive_sampling",
        "residual_adaptive",
        candidate_pool=512,
        power=1,
        uniform_mix=0.1,
        epsilon=1e-12,
        resample_every=50,
    ),
    op.method(
        "adaptive_weights",
        "uncertainty_weighting",
        lr=0.001,
        min_log_variance=-10,
        max_log_variance=10,
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs"

result = op.solve(
    PROBLEM,
    methods=METHODS,
    steps=STEPS,
    seed=SEED,
    training={"log_every": LOG_EVERY},
    output_root=OUTPUT_ROOT,
)

print("训练完成")
print("结果目录：", result.output_dir)