"""Metadata for the five-axis method catalogue.

The executable registries remain the source of truth.  This module only
describes families that those registries can construct.  Qualified names such
as ``operator:fno`` and ``feature:fourier`` identify the registry that owns
the implementation.  The repository does not include a paper whose section
4.4 defines this list, so the strings below describe the code that runs.
"""

from __future__ import annotations

from typing import Any

from .experiment import default_component_registry
from .operator_protocol import default_operator_registry


METHOD_ECOSYSTEM_AXES: tuple[dict[str, Any], ...] = (
    {
        "kind": "model",
        "title": "模型 Model",
        "description": "选择点式场网络、约束包装、区域分解或神经算子骨干。",
        "families": (
            ("pointwise_mlp", "Pointwise MLP", "MLP", "solve() 的全连接骨干", ("mlp",)),
            ("modified_mlp", "Modified MLP", "门控 MLP", "两个编码分支与门控混合", ("modifiedmlp",)),
            ("spectral_embedding", "谱嵌入", "Fourier features", "feature 注册表中的固定随机傅里叶特征", ("feature:fourier",)),
            ("hard_constraints", "硬约束", "距离函数输出变换", "constraint 注册表；当前只精确处理已支持的区间 Dirichlet", ("constraint:hard",)),
            ("domain_decomposition", "区域分解", "两子域网络", "一维非均质 Poisson 的解连续与通量连续", ("domaindecomposition",)),
            ("branch_trunk", "Branch–trunk 算子", "DeepONet", "operator 训练入口，模型名不进入 solve() 的 model.name", ("operator:deeponet",)),
            ("graph_operators", "图消息传递", "全配点 RBF 聚合", "稠密径向基消息传递；multipole 项是全局均值通道", ("gno", "multipolegno")),
            ("spectral_operators", "谱算子与可分基", "FNO / 可分坐标基", "fno 走算子训练；low_rank_operator 是正弦余弦可分坐标基", ("operator:fno", "lowrankoperator")),
            ("emerging_bases", "新兴基函数", "Physics-informed KAN", "每个输入维一组可训练高斯一元基", ("physicsinformedkan",)),
        ),
    },
    {
        "kind": "sampler",
        "title": "采样 Sampling",
        "description": "控制残差、边界、数据与时间切片的训练位置。",
        "families": (
            ("non_adaptive", "非自适应", "Uniform / grid / LHS / Sobol", "均匀随机、截断张量积网格、拉丁超立方，以及不依赖 SciPy 的未加扰 Sobol", ("uniform", "grid", "lhs", "qmc")),
            ("residual_adaptive", "残差自适应", "RAR / RAD / importance sampling", "rar 追加高残差点并保留最新一批；另有已实现的 RAD 与重要性采样", ("rar", "residualadaptive", "importancesampling")),
            ("gradient_adaptive", "梯度自适应", "残差或梯度移动", "保留固定点，并按残差或空间梯度移动其余配点", ("adaptivemovement",)),
            ("uncertainty_adaptive", "不确定性自适应", "参数扰动分歧", "对当前参数加固定幅度高斯噪声后的预测标准差", ("uncertainty",)),
            ("learned_adaptive", "学习式自适应", "残差分数密度拟合", "用少量 Adam 步把小型网络拟合到候选残差分数，再按拟合分数抽样", ("das",)),
            ("temporal_curriculum", "时间课程", "顺序增长时间窗", "按给定窗口端点逐步扩展时间域", ("timewindow",)),
        ),
    },
    {
        "kind": "loss",
        "title": "损失 Loss",
        "description": "组装残差、边界、初值与数据项，并控制动态平衡。",
        "families": (
            ("fixed_composite", "固定权重组合", "手调常数权重", "各项显式常数权重求和", ("fixed",)),
            ("gradient_balanced", "梯度平衡", "平均绝对梯度", "用各项损失对参数梯度的平均绝对值做动量平衡", ("gradientbalanced",)),
            ("ntk_balanced", "梯度范数平衡", "损失梯度平方和", "注册名 ntk_balanced 与别名 ntk_weighting 使用同一计算", ("ntkbalanced",)),
            ("uncertainty_balanced", "不确定性平衡", "对数方差 / 梯度方差", "uncertainty 学习对数方差；inverse_dirichlet 用参数梯度方差的倒数做动量平衡", ("uncertainty", "inversedirichlet")),
            ("pointwise_adaptive", "逐点自适应", "逐点正权重", "每个配点一个可学习正权重", ("pointwiseselfadaptive",)),
            ("multi_objective", "多目标调度", "ReLoBRaLo", "按相对初值和相对上一步的变化更新权重", ("relobralo",)),
            ("causal_curriculum", "因果损失", "时间排序指数权重", "时空问题按时间坐标排序后做指数累积权重；静态问题会拒绝", ("causal",)),
        ),
    },
    {
        "kind": "optimizer",
        "title": "优化 Optimization",
        "description": "选择更新阶数、学习率计划、预算与多阶段协议。",
        "families": (
            ("first_order", "一阶方法", "Adam / AdamW", "一阶矩估计", ("adam", "adamw")),
            ("quasi_newton", "拟牛顿", "L-BFGS", "全批量 L-BFGS", ("lbfgs",)),
            ("second_order", "对角曲率归一化", "梯度平方预处理", "注册名 gauss_newton、natural_gradient 由 solve() 的 optimizer.name 调用", ("gaussnewton", "naturalgradient")),
            ("hybrid_multistage", "混合多阶段", "Adam → L-BFGS", "按配置步数从 Adam 切换到 L-BFGS", ("adamlbfgs",)),
        ),
    },
    {
        "kind": "derivative",
        "title": "微分 Differentiation",
        "description": "决定 PDE 残差中的空间与时空导数如何形成。",
        "families": (
            ("coordinate_ad", "坐标自动微分", "反向自动微分", "solve() 强形式残差使用的坐标导数", ("autodiff",)),
            ("numerical", "数值微分", "周期中心差分", "形状 (batch, channel, *space) 的周期网格差分，不替换 solve() 的坐标自动微分", ("finitedifference",)),
            ("spectral_grid", "谱网格导数", "傅里叶乘子", "周期网格上的谱导数，供网格算子调用", ("grid",)),
            ("discrete_grid", "离散网格算子", "周期差分与面平均", "注册名 finite_volume 与 finitedifference 都是周期网格模板", ("finitedifference", "finitevolume")),
            ("weak_variational", "弱式 / 变分残差", "分部积分求积", "integrate_by_parts 可单独调用；solve() 仍计算强形式残差", ("weakvariational",)),
            ("hybrid_ad_numerical", "AD 与网格差分", "两条独立导数路径", "grad 使用坐标自动微分，partial 使用周期网格差分", ("hybridadnumerical",)),
        ),
    },
)


# Exact formulas and call paths.  These strings are shown on the configurator
# card and are the contract tests lock.  They do not rename the registry ids.
FAMILY_BOUNDARIES: dict[str, str] = {
    "pointwise_mlp": "solve() 通过 model.name=mlp 构造 PINNModel。",
    "modified_mlp": "solve() 通过 model.name=modified_mlp 构造门控网络。",
    "spectral_embedding": "feature:fourier 由模型的 feature 配置调用，不是 model.name。",
    "hard_constraints": "constraint:hard 由模型的 output_transform 配置调用，只处理已支持的区间 Dirichlet。",
    "domain_decomposition": "solve() 通过 model.name=domain_decomposition 构造两子域网络，仅一维非均质 Poisson。",
    "branch_trunk": "operator:deeponet 由 train_operator 调用，solve() 的 model.name 不接受它。",
    "graph_operators": "solve() 通过 model.name=gno 或 multipole_gno 调用。gno 是全配点径向基消息传递；multipole_gno 额外加上全局均值通道。",
    "spectral_operators": "operator:fno 由 train_operator 调用。model:low_rank_operator 由 solve() 调用，计算 sin(Wx) 与 cos(Vx) 的逐点乘积。",
    "emerging_bases": "solve() 通过 model.name=physics_informed_kan 调用，每个输入维使用可训练高斯一元基。",
    "non_adaptive": "solve() 通过 sampling.name 调用。qmc 是未加扰 Sobol 序列。",
    "residual_adaptive": "solve() 通过 sampling.name 调用。rar 按残差追加新点并保留最新一批。",
    "gradient_adaptive": "solve() 通过 sampling.name=adaptive_movement 调用，按残差或空间梯度移动部分配点。",
    "uncertainty_adaptive": "solve() 通过 sampling.name=uncertainty 调用，对当前参数加 1e-3 高斯噪声后取预测标准差。",
    "learned_adaptive": "solve() 通过 sampling.name=das 调用，用少量 Adam 步把小网络拟合到候选残差分数。",
    "temporal_curriculum": "solve() 通过 sampling.name=time_window 调用，按 window_ends 扩展时间窗。",
    "fixed_composite": "solve() 通过 loss.weighting=fixed 调用，各项乘常数权重后相加。",
    "gradient_balanced": "solve() 通过 loss.weighting=gradient_balanced 或别名 learning_rate_annealing 调用。统计量是该损失项对参数梯度的平均绝对值。",
    "ntk_balanced": "solve() 通过 loss.weighting=ntk_balanced 或别名 ntk_weighting 调用。统计量是该损失项对参数梯度的平方和，权重是各统计量均值除以该项统计量后再做动量平均。",
    "uncertainty_balanced": "solve() 通过 loss.weighting 调用。uncertainty 学习对数方差；inverse_dirichlet 的统计量是参数梯度的方差。",
    "pointwise_adaptive": "solve() 通过 loss.weighting=pointwise_self_adaptive 调用，每个配点一个正权重。",
    "multi_objective": "solve() 通过 loss.weighting=relobralo 调用。",
    "causal_curriculum": "solve() 通过 loss.weighting=causal 调用。时空批次按最后一列时间排序后做指数累积权重；静态域会拒绝。",
    "first_order": "solve() 通过 optimizer.name=adam 或 adamw 调用 PyTorch 的对应优化器。",
    "quasi_newton": "solve() 通过 optimizer.name=lbfgs 调用全批量 L-BFGS。",
    "second_order": "solve() 通过 optimizer.name=gauss_newton 或别名 gn 调用时，一步是 parameter -= lr * grad / (|grad| + damping)。optimizer.name=natural_gradient 或别名 ngd 使用梯度平方的指数滑动 v，一步是 parameter -= lr * grad / (sqrt(v) + damping)。",
    "hybrid_multistage": "solve() 通过 optimizer.name=adam_lbfgs 按阶段步数从 Adam 切到 L-BFGS。",
    "coordinate_ad": "solve() 默认 differentiator.mode=autodiff，强形式残差的坐标 grad/hessian 由该后端计算。",
    "numerical": "derivative:finitedifference 的 partial 是周期中心差分。强形式 solve() 发现它没有坐标 grad/hessian 后拒绝训练；网格算子仍单独调用 partial。",
    "spectral_grid": "derivative:grid 的 partial 是周期傅里叶乘子。强形式 solve() 发现它没有坐标 grad/hessian 后拒绝训练。",
    "discrete_grid": "注册名 finite_volume 与别名 fv 的一阶 partial 等于 (u[i+1] - u[i-1]) / (2 * spacing)，端点用 torch.roll 周期卷绕。入口是 registry.create('derivative', 'finite_volume').partial(field, axis)。强形式 solve() 在缺少坐标 grad/hessian 后拒绝训练，因此不用它形成残差。",
    "weak_variational": "solve() 可通过 differentiator.mode=weak_variational 安装该后端。其 grad/hessian 仍是坐标自动微分；integrate_by_parts 计算 -sum(flux * test_gradient * weights)，强形式残差不调用它，导数阶数不变。",
    "hybrid_ad_numerical": "solve() 可通过 differentiator.mode=hybrid_ad_numerical 安装该后端。强形式使用的 grad 仍是坐标自动微分；partial 是周期网格差分，两条路径不相加。",
}


def method_ecosystem_payload() -> dict[str, Any]:
    """Return the five axes annotated from the live executable registries."""

    components = default_component_registry()
    operators = default_operator_registry()
    axes: list[dict[str, Any]] = []
    for axis in METHOD_ECOSYSTEM_AXES:
        families: list[dict[str, Any]] = []
        for slug, title, forms, setting, implementations in axis["families"]:
            available: list[str] = []
            for implementation in implementations:
                if implementation.startswith("operator:"):
                    name = implementation.split(":", 1)[1]
                    try:
                        spec = operators.resolve(name)
                    except ValueError:
                        continue
                    if spec.selectable:
                        available.append(f"operator:{spec.name}")
                elif ":" in implementation:
                    kind, name = implementation.split(":", 1)
                    if components.contains(kind, name):
                        available.append(
                            f"{kind}:{components.canonical_name(kind, name)}"
                        )
                elif components.contains(axis["kind"], implementation):
                    available.append(
                        f"{axis['kind']}:{components.canonical_name(axis['kind'], implementation)}"
                    )
            if not available:
                raise RuntimeError(
                    f"Method ecosystem family {axis['kind']}:{slug} has no built-in implementation."
                )
            boundary = FAMILY_BOUNDARIES.get(slug)
            if not boundary:
                raise RuntimeError(f"Method family {axis['kind']}:{slug} has no algorithm boundary.")
            families.append({
                "slug": slug,
                "title": title,
                "forms": forms,
                "setting": setting,
                "boundary": boundary,
                "status": "built_in",
                "implementations": list(dict.fromkeys(available)),
                "documentation": f"method_ecosystem/{axis['kind']}/{slug}.html",
            })
        axes.append({
            "kind": axis["kind"],
            "title": axis["title"],
            "description": axis["description"],
            "families": families,
        })
    return {
        "axes": axes,
        "extension_kinds": [axis["kind"] for axis in axes],
        "configurator": method_ecosystem_configurator_contract(),
    }


def method_ecosystem_configurator_contract() -> dict[str, Any]:
    """Return PINN component choices that can be emitted as solve overrides."""

    def choices(*items: tuple[str, str, dict[str, Any] | None]) -> list[dict[str, Any]]:
        return [
            {"value": value, "label": label, "config": dict(config or {})}
            for value, label, config in items
        ]

    controls = [
        {
            "key": "model",
            "axis": "model",
            "label": "模型骨干",
            "selection_mode": "exclusive",
            "target": ["model"],
            "choices": choices(
                ("mlp", "Pointwise MLP", {"hidden_layers": [64, 64, 64], "activation": "tanh", "bias": True}),
                ("modifiedmlp", "Modified MLP", {"hidden_layers": [64, 64, 64], "activation": "tanh", "bias": True}),
                ("gno", "全配点 RBF 消息传递", {"hidden_layers": [48, 48, 48], "length_scale": 1.0}),
                ("multipolegno", "RBF 消息传递加全局均值", {"hidden_layers": [48, 48, 48], "length_scale": 1.0}),
                ("lowrankoperator", "正弦余弦可分坐标基", {"rank": 32}),
                ("physicsinformedkan", "可训练高斯一元基", {"grid_size": 12, "width": 32}),
                ("domaindecomposition", "两子域网络（非均质问题）", {"hidden_layers": [32, 32, 32], "activation": "tanh", "bias": True}),
            ),
        },
        {
            "key": "feature",
            "axis": "model",
            "label": "输入表示",
            "selection_mode": "optional_composable",
            "optional": True,
            "target": ["model", "feature"],
            "choices": choices(
                ("identity", "原始坐标", None),
                ("fourier", "Fourier features", {"num_frequencies": 16, "scale": 1.0, "seed": 0, "include_input": False, "trainable": False}),
            ),
        },
        {
            "key": "constraint",
            "axis": "model",
            "label": "输出约束",
            "selection_mode": "optional_composable",
            "optional": True,
            "target": ["model", "output_transform"],
            "choices": choices(
                ("identity", "软约束", None),
                ("hard", "硬边界约束", None),
            ),
        },
        {
            "key": "sampler",
            "axis": "sampler",
            "label": "采样",
            "selection_mode": "exclusive",
            "target": ["sampling"],
            "choices": choices(
                ("uniform", "Uniform", {"interior": 128, "boundary": 32, "initial": 32}),
                ("grid", "规则网格", {"interior": 128, "boundary": 32, "initial": 32}),
                ("lhs", "Latin hypercube", {"interior": 128, "boundary": 32, "initial": 32}),
                ("qmc", "未加扰 Sobol", {"interior": 128, "boundary": 32, "initial": 32}),
                ("rar", "RAR", {"interior": 128, "boundary": 32, "initial": 32, "candidate_pool": 512, "enrich": 16}),
                ("residualadaptive", "RAD", {"interior": 128, "boundary": 32, "initial": 32}),
                ("importancesampling", "Importance sampling", {"interior": 128, "boundary": 32, "initial": 32}),
                ("adaptivemovement", "残差或梯度移动", {"interior": 128, "boundary": 32, "initial": 32, "indicator": "solution_gradient"}),
                ("uncertainty", "参数扰动分歧", {"interior": 128, "boundary": 32, "initial": 32, "candidate_pool": 512, "ensemble_size": 4}),
                ("das", "残差分数密度拟合", {"interior": 128, "boundary": 32, "initial": 32, "candidate_pool": 512, "density_steps": 12}),
                ("timewindow", "顺序时间窗（动态问题）", {"window_ends": [0.25, 0.5, 0.75, 1.0]}),
            ),
        },
        {
            "key": "loss",
            "axis": "loss",
            "label": "损失权重",
            "selection_mode": "exclusive",
            "target": ["loss", "weighting"],
            "choices": choices(
                ("fixed", "固定权重", None),
                ("gradientbalanced", "梯度平均绝对值", {"momentum": 0.9}),
                ("ntkbalanced", "损失梯度平方和", {"momentum": 0.9}),
                ("uncertainty", "可学习对数方差", None),
                ("inversedirichlet", "参数梯度方差", {"momentum": 0.9}),
                ("selfadaptive", "项级自适应权重", None),
                ("pointwiseselfadaptive", "逐点正权重", None),
                ("relobralo", "ReLoBRaLo", None),
                ("magnitudepower", "损失幂压缩", None),
                ("causal", "时间排序因果权重", {"epsilon": 1.0}),
            ),
        },
        {
            "key": "optimizer",
            "axis": "optimizer",
            "label": "优化器",
            "selection_mode": "exclusive",
            "target": ["optimizer"],
            "choices": choices(
                ("adam", "Adam", {"lr": 0.001}),
                ("adamw", "AdamW", {"lr": 0.001}),
                ("lbfgs", "L-BFGS", {"lr": 1.0, "max_iter": 20}),
                ("gaussnewton", "对角梯度归一化", {"lr": 0.01, "damping": 1e-8}),
                ("naturalgradient", "梯度平方滑动预处理", {"lr": 0.01, "damping": 1e-8, "beta": 0.95}),
                ("adamlbfgs", "Adam → L-BFGS", {"stages": [
                    {"name": "adam", "steps": 500, "lr": 0.001},
                    {"name": "lbfgs", "steps": 50, "lr": 1.0, "max_iter": 20},
                ]}),
            ),
        },
        {
            "key": "derivative",
            "axis": "derivative",
            "label": "微分路径",
            "selection_mode": "exclusive",
            "target": ["differentiator"],
            "choices": choices(
                ("autodiff", "坐标自动微分", None),
                ("weakvariational", "坐标自动微分加分部积分助手", None),
                ("hybridadnumerical", "坐标自动微分与网格差分并列", {"spacing": 1.0}),
            ),
        },
    ]
    # Keep one five-axis selector for both solver paths while showing only
    # choices that the selected path can serialize into its local script.
    for control in controls:
        for choice in control["choices"]:
            choice["paths"] = ["pinn"]
    model = next(control for control in controls if control["key"] == "model")
    model["choices"].extend(
        [
            {"value": "deeponet", "label": "DeepONet", "config": {}, "paths": ["operator"]},
            {"value": "fno", "label": "FNO", "config": {}, "paths": ["operator"]},
        ]
    )
    for choice in next(control for control in controls if control["key"] == "sampler")["choices"]:
        choice["paths"] = ["pinn", "operator"]
    optimizer_control = next(control for control in controls if control["key"] == "optimizer")
    operator_optimizers = {choice["value"] for choice in optimizer_control["choices"]}
    for key, operator_values in (("loss", {"fixed"}), ("optimizer", operator_optimizers)):
        for choice in next(control for control in controls if control["key"] == key)["choices"]:
            if choice["value"] in operator_values:
                choice["paths"] = ["pinn", "operator"]
    derivative = next(control for control in controls if control["key"] == "derivative")
    derivative["choices"].extend(
        [
            {"value": "finitedifference", "label": "周期中心差分", "config": {}, "paths": ["operator"]},
            {"value": "grid", "label": "谱网格导数", "config": {}, "paths": ["operator"]},
            {"value": "finitevolume", "label": "有限体积周期模板", "config": {}, "paths": ["operator"]},
        ]
    )
    return {
        "controls": controls,
        "defaults": {
            "model": "mlp",
            "feature": "identity",
            "constraint": "identity",
            "sampler": "uniform",
            "loss": "fixed",
            "optimizer": "adam",
            "derivative": "autodiff",
        },
    }


__all__ = [
    "METHOD_ECOSYSTEM_AXES",
    "method_ecosystem_configurator_contract",
    "method_ecosystem_payload",
]
