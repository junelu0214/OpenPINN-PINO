# OpenPINN 代码教程

三条主线是 `04_pinn_burgers.ipynb`（PINN 1D Burgers）、
`05_pinn_poisson_2d.ipynb`（PINN 2D Poisson）和
`07_operator_fno_physics_only.ipynb`（Operator 2D Poisson，FNO × physics-only）。
它们包含物理设置、已提交的演示损失/预测图，以及单独标明的 CPU smoke。
演示文件在 `artifacts/m19/{burgers,poisson_2d_pinn,fno_physics_only}/`，只有
PNG 和指标，没有 checkpoint。`05_pinn_plate_with_hole.ipynb` 保留为不进入侧栏的
带孔板页面。其余 notebooks 覆盖安装、读取产物、一维 PINN Poisson、DeepONet
data-only、FNO hybrid 与自定义 PDE/模式切换。

所有训练都使用固定随机种子、CPU 和小步数，只验证调用、产物与可视化链路，不构成
精度或性能结论。浏览器配置器仍可独立下载 `.py`；notebook 和配置器都只在本地训练。

## 工作目录与安装

从仓库根目录安装；``docs`` extra 已包含执行 notebooks 所需的 kernel 与
``nbconvert``：

```bash
python -m pip install -e ".[test,docs]"
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 \
  examples/tutorials/04_pinn_burgers.ipynb
```

notebook 会自动定位包含 `pyproject.toml` 的仓库根目录。需要交互界面时，可以使用
本机已有的 JupyterLab 打开同一目录；它不是构建文档或复现 outputs 的必要依赖。
Sphinx/nbsphinx 生成 HTML 还需要系统 `pandoc`；CI 会安装它，本地构建文档前也请
通过系统包管理器安装。

## 刷新并提交输出

仓库内已经执行的 outputs 是 Sphinx 文档的来源。Sphinx 使用
`nbsphinx_execute = "never"`，CI 不会重新训练 notebooks。修改教程代码或相关公开
行为后，维护者应在 CPU 环境从仓库根目录刷新输出：

```bash
for notebook in examples/tutorials/*.ipynb; do
  python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=180 "$notebook"
done
```

检查每个 notebook 的输出不含本机秘密或无关绝对路径，运行 notebook 验证测试与
Sphinx `-W` 构建，再把源单元和新 outputs 一起提交。仅改文字且不影响代码时无需刷新
训练输出。

不要添加 Binder、Colab 或 JupyterLite 配置；站点只展示已提交结果并提供下载。
