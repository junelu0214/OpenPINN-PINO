"""Twin-tunnel 2D model: choose the problem and method with short names."""

from pathlib import Path

import openpinn as op


PROJECT_ROOT = Path(__file__).resolve().parents[1]

result = op.solve(
    "tunnel_model1",       # 双隧道二维模型
    method="adaptive_weights",  # 方法大类
    variant="relobralo",        # 具体方法简称
    steps=5,                     # 先检查链路；论文报告预算为 20,000
    output_root=PROJECT_ROOT / "runs",
)
print(result.output_dir)
