"""User-facing method selections and parameter metadata.

This module is the single catalogue used by the short Python API and the
documentation configurator.  A parameter listed here is editable through
``op.method(...)``; unknown names fail before training starts.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MethodSelection:
    """One concrete method selected for a composed OpenPINN run."""

    method: str
    variant: str
    options: Mapping[str, Any] = field(default_factory=dict)


def _parameter(
    label: str,
    kind: str,
    default: Any = None,
    *,
    description: str,
    effect: str,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: Sequence[Any] = (),
    required: bool = False,
    placeholder: str = "",
) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "default": default,
        "description": description,
        "effect": effect,
        "minimum": minimum,
        "maximum": maximum,
        "choices": list(choices),
        "required": required,
        "placeholder": placeholder,
    }


METHOD_PARAMETER_SCHEMAS: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
    ("baseline", "plain_pinn"): {},
    ("modified_mlp", "gated_mlp"): {},
    ("adaptive_activation", "adaptive_tanh"): {
        "initial_slope": _parameter(
            "初始激活斜率",
            "number",
            1.0,
            description="每个隐藏层可训练 tanh 斜率的初始值。",
            effect="调大使初始激活变化更陡，可能更快表示局部变化，但过大容易梯度不稳；调小更平滑，但可能学习较慢。",
            minimum=1e-12,
        ),
    },
    ("fourier", "random_fourier"): {
        "num_frequencies": _parameter(
            "频率数量",
            "integer",
            16,
            description="每个输入维度使用的随机 Fourier 频率数。",
            effect="调大会增加特征维度、参数量和高频表达能力，也会增加时间与过拟合风险；调小更省算力，但可能漏掉高频。",
            minimum=1,
        ),
        "scale": _parameter(
            "频率尺度",
            "number",
            1.0,
            description="随机频率矩阵的尺度；越大越强调高频。",
            effect="调大偏向更短波长，适合高频解但可能让优化变难；调小偏向平滑、低频结构。",
            minimum=1e-12,
        ),
        "seed": _parameter(
            "特征随机种子",
            "integer",
            29,
            description="固定随机 Fourier 特征的随机种子。",
            effect="改变它会换一组随机频率，不代表数值越大越好；正式比较应固定种子或报告多种子结果。",
            minimum=0,
        ),
        "include_input": _parameter(
            "保留原始坐标",
            "boolean",
            True,
            description="是否把原始坐标与 Fourier 特征拼接。",
            effect="设为 True 可保留低频和坐标直连信息但增加输入维度；False 更精简，但完全依赖 Fourier 特征。",
        ),
        "trainable": _parameter(
            "训练频率矩阵",
            "boolean",
            False,
            description="是否把随机频率矩阵也作为可训练参数。",
            effect="True 增加可训练自由度，可能适应问题频谱但更慢、更易不稳；False 固定特征，复现更简单。",
        ),
    },
    ("hard_boundary", "exact_output_transform"): {},
    ("adaptive_weights", "self_adaptive"): {
        "lr": _parameter(
            "权重学习率",
            "number",
            0.01,
            description="损失权重梯度上升时使用的学习率。",
            effect="调大让权重反应更快，但可能振荡或迅速顶到上限；调小更稳定，但难以在有限步数内明显调整。",
            minimum=1e-12,
        ),
        "min_weight": _parameter(
            "最小权重",
            "number",
            1e-6,
            description="自适应权重允许达到的下界。",
            effect="调大可防止某项被忽略，但可能压制其他任务；调小允许更强降权，也可能让某项几乎失去作用。",
            minimum=0.0,
        ),
        "max_weight": _parameter(
            "最大权重",
            "number",
            100.0,
            description="自适应权重允许达到的上界。",
            effect="调大允许更强地关注困难项，但可能造成梯度爆炸；调小更稳，却可能限制自适应能力。",
            minimum=1e-12,
        ),
    },
    ("adaptive_weights", "uncertainty_weighting"): {
        "lr": _parameter(
            "对数方差学习率",
            "number",
            0.001,
            description="学习各损失项对数方差时的学习率。",
            effect="调大使任务权重变化更快但更易波动；调小变化更平滑，但适应速度更慢。",
            minimum=1e-12,
        ),
        "min_log_variance": _parameter(
            "最小对数方差",
            "number",
            -10.0,
            description="对数方差参数的下界。",
            effect="设得更小允许更大的实际权重 exp(-s)，也增加数值过大风险；提高下界会限制最大权重。",
        ),
        "max_log_variance": _parameter(
            "最大对数方差",
            "number",
            10.0,
            description="对数方差参数的上界。",
            effect="设得更大允许把高不确定项降得更低，也可能近似忽略该项；降低上界可避免过度降权。",
        ),
    },
    ("adaptive_weights", "relobralo"): {
        "alpha": _parameter(
            "历史平滑系数",
            "number",
            0.999,
            description="当前权重与历史权重之间的指数平滑系数。",
            effect="接近 1 时权重更平滑、记忆更长但响应慢；减小时响应更快，也更容易抖动。",
            minimum=0.0,
            maximum=1.0,
        ),
        "temperature": _parameter(
            "温度",
            "number",
            0.1,
            description="控制相对损失 softmax 分布的尖锐程度。",
            effect="调小会放大损失差异、权重更尖锐且可能不稳；调大使各项权重更接近，适应性更弱。",
            minimum=1e-12,
        ),
        "expected_lookback": _parameter(
            "回看概率期望",
            "number",
            0.999,
            description="使用上一步损失而非初始损失的期望概率。",
            effect="接近 1 时主要关注短期变化；调小会更频繁回看初始损失，增加长期参照和随机性。",
            minimum=0.0,
            maximum=1.0,
        ),
        "epsilon": _parameter(
            "数值稳定量",
            "number",
            1e-12,
            description="计算相对损失时避免除零的极小正数。",
            effect="通常保持默认；过大可能扭曲很小的损失比值，过小在接近零时可能降低数值保护。",
            minimum=0.0,
        ),
        "seed": _parameter(
            "方法随机种子",
            "integer",
            0,
            description="随机回看过程的随机种子。",
            effect="改变它会改变随机回看序列，不存在越大越好；公平比较时应固定或采用多随机种子。",
            minimum=0,
        ),
    },
    ("adaptive_weights", "magnitude_power"): {
        "roots": _parameter(
            "各损失开方次数",
            "number_map",
            {"pde": 3.0, "boundary": 2.0, "initial": 2.0},
            description="损失名称到幂根次数的映射，例如 PDE 使用三次方根。",
            effect="某项的根次数调大，会更强地压缩大损失的数量级；调小则更接近原损失。必须使用实际损失项名称。",
        ),
        "default_root": _parameter(
            "未列出损失的开方次数",
            "number",
            1.0,
            description="没有出现在 roots 中的损失项使用的根次数。",
            effect="调大对未列出的损失进行更强压缩；设为 1 表示保持原损失。不要设为零或负数。",
            minimum=1e-12,
        ),
        "epsilon": _parameter(
            "数值稳定量",
            "number",
            1e-12,
            description="幂变换时使用的数值稳定量。",
            effect="通常保持默认；过大将改变小损失，过小可能减弱零附近的数值保护。",
            minimum=0.0,
        ),
    },
    ("adaptive_weights", "pointwise_self_adaptive"): {
        "lr": _parameter(
            "逐点权重学习率",
            "number",
            0.01,
            description="逐训练点掩码梯度上升时使用的学习率。",
            effect="调大让难点权重增长更快，但更易振荡或饱和；调小更稳，但需要更多训练步才有明显效果。",
            minimum=1e-12,
        ),
        "min_weight": _parameter(
            "最小逐点权重",
            "number",
            1e-6,
            description="逐点权重允许达到的下界。",
            effect="调大能防止任何训练点被忽略，但降低稀疏选择能力；调小允许弱化易点，也可能让其几乎无贡献。",
            minimum=0.0,
        ),
        "max_weight": _parameter(
            "最大逐点权重",
            "number",
            100.0,
            description="逐点权重允许达到的上界。",
            effect="调大允许极难点获得更高关注，但可能产生大梯度；调小更稳，却可能压制难点权重。",
            minimum=1e-12,
        ),
    },
    ("adaptive_sampling", "residual_adaptive"): {
        "candidate_pool": _parameter(
            "候选点数量",
            "integer",
            512,
            description="每次重采样前用于评估残差的候选内部点数量。",
            effect="调大覆盖更充分、找到难点的机会更高，但每次重采样更耗时和内存；调小更快但选择更粗糙。",
            minimum=1,
        ),
        "power": _parameter(
            "残差幂指数",
            "number",
            1.0,
            description="残差转为抽样概率时使用的幂指数。",
            effect="调大更集中采样高残差区域，可能丢失全局覆盖；调小使分布更接近均匀，适应性减弱。",
            minimum=0.0,
        ),
        "uniform_mix": _parameter(
            "均匀采样混合比例",
            "number",
            0.1,
            description="保留均匀覆盖的概率比例。",
            effect="调大增强全域探索但减少难点聚焦；调小更集中高残差点，也更容易遗漏其他区域。",
            minimum=0.0,
            maximum=1.0,
        ),
        "epsilon": _parameter(
            "数值稳定量",
            "number",
            1e-12,
            description="残差转概率时避免全零的极小正数。",
            effect="通常保持默认；调大使概率更均匀，过大可能淹没残差差异；过小在全零附近保护较弱。",
            minimum=0.0,
        ),
        "resample_every": _parameter(
            "重采样间隔",
            "integer",
            50,
            description="每隔多少个外层训练步更新一次内部点。",
            effect="调小会更频繁追踪当前误差但增加采样开销和波动；调大更省算力，但训练点可能更新不及时。",
            minimum=1,
        ),
    },
    ("adaptive_sampling", "residual_movement"): {
        "candidate_pool": _parameter(
            "候选点数量", "integer", 512,
            description="每次移动配点前用于评估指标的候选点数量。",
            effect="调大能更充分搜索高残差位置，但增加移动时的计算和内存；调小更快但搜索范围更粗。",
            minimum=1,
        ),
        "movable_fraction": _parameter(
            "可移动点比例", "number", 0.5,
            description="每次更新中允许移动的内部点比例。",
            effect="调大让配点更积极追随残差，但全局固定覆盖减少；调小保留更多固定点，适应速度变慢。",
            minimum=0.0, maximum=1.0,
        ),
        "power": _parameter(
            "残差幂指数", "number", 1.0,
            description="残差转为移动概率时使用的幂指数。",
            effect="调大更偏向最高残差位置但可能过度集中；调小更均匀、覆盖更广。",
            minimum=0.0,
        ),
        "epsilon": _parameter(
            "数值稳定量", "number", 1e-12,
            description="残差转概率时使用的极小正数。",
            effect="通常保持默认；调大使移动概率更均匀，过大可能掩盖残差差异。",
            minimum=0.0,
        ),
        "resample_every": _parameter(
            "移动间隔", "integer", 50,
            description="每隔多少个外层训练步移动一次配点。",
            effect="调小会更频繁移动、开销和波动更大；调大更省算力，但点可能长期停留在旧位置。",
            minimum=1,
        ),
    },
    ("adaptive_sampling", "gradient_movement"): {
        "candidate_pool": _parameter(
            "候选点数量", "integer", 512,
            description="每次移动配点前用于评估指标的候选点数量。",
            effect="调大能更充分搜索高梯度位置，但增加移动时的计算和内存；调小更快但搜索更粗。",
            minimum=1,
        ),
        "movable_fraction": _parameter(
            "可移动点比例", "number", 0.5,
            description="每次更新中允许移动的内部点比例。",
            effect="调大让更多点追随解梯度，但固定全域覆盖减少；调小更保守，也可能错过陡变区。",
            minimum=0.0, maximum=1.0,
        ),
        "power": _parameter(
            "梯度幂指数", "number", 1.0,
            description="解梯度转为移动概率时使用的幂指数。",
            effect="调大更聚焦最陡区域但可能过度集中；调小分布更均匀，梯度导向更弱。",
            minimum=0.0,
        ),
        "epsilon": _parameter(
            "数值稳定量", "number", 1e-12,
            description="解梯度转概率时使用的极小正数。",
            effect="通常保持默认；调大使概率更平滑，过大可能掩盖梯度差异。",
            minimum=0.0,
        ),
        "resample_every": _parameter(
            "移动间隔", "integer", 50,
            description="每隔多少个外层训练步移动一次配点。",
            effect="调小会更频繁更新且更耗时；调大更省算力，但点对当前解变化的响应更慢。",
            minimum=1,
        ),
    },
    ("adaptive_sampling", "importance_sampling"): {
        "candidate_pool": _parameter(
            "候选点数量", "integer", 512,
            description="用于估计逐点损失及抽样概率的候选点数量。",
            effect="调大能更好估计重要性分布，但增加每次评估成本；调小更快但概率估计噪声更大。",
            minimum=1,
        ),
        "epsilon": _parameter(
            "数值稳定量", "number", 1e-12,
            description="计算抽样概率和逆概率权重时使用的极小正数。",
            effect="调大可压低极端逆概率权重、提高稳定性，但会削弱重要性差异；过小可能产生很大的修正权重。",
            minimum=0.0,
        ),
        "resample_every": _parameter(
            "重采样间隔", "integer", 50,
            description="每隔多少个外层训练步重新执行重要性采样。",
            effect="调小能更快跟随损失分布但增加成本和方差；调大更省算力，但采样概率可能过时。",
            minimum=1,
        ),
    },
    ("adam_lbfgs", "adam_then_lbfgs"): {
        "adam_lr": _parameter(
            "Adam 学习率", "number", 1e-3,
            description="第一阶段 Adam 优化器的学习率。",
            effect="调大前期下降可能更快但更易震荡或发散；调小更稳，但需要更多 Adam 步。",
            minimum=1e-12,
        ),
        "lbfgs_lr": _parameter(
            "L-BFGS 学习率", "number", 1.0,
            description="第二阶段 L-BFGS 优化器的学习率。",
            effect="调大单次更新更激进，可能降低更快也可能不稳；调小更保守，收尾速度可能变慢。",
            minimum=1e-12,
        ),
        "lbfgs_fraction": _parameter(
            "L-BFGS 步数比例", "number", 0.02,
            description="总外层训练步数中分给 L-BFGS 的比例。",
            effect="调大给二阶收尾更多外层步、通常更耗时；调小把预算更多留给 Adam，可能收尾不足。",
            minimum=1e-12, maximum=0.999999,
        ),
        "lbfgs_max_steps": _parameter(
            "L-BFGS 最大外层步数", "integer", 20,
            description="限制 L-BFGS 使用的外层训练步数。",
            effect="调大允许更充分收尾但增加 closure 调用与时间；调小限制成本，也可能提前停止。",
            minimum=1,
        ),
        "lbfgs_max_iter": _parameter(
            "每步最大内部迭代", "integer", 10,
            description="每个 L-BFGS 外层步允许的最大内部迭代次数。",
            effect="调大每个外层步可能更充分，但时间和显存成本上升；调小更快，单步优化可能不充分。",
            minimum=1,
        ),
        "history_size": _parameter(
            "历史向量数量", "integer", 50,
            description="L-BFGS 保存的历史校正向量数量。",
            effect="调大可利用更多曲率历史但增加内存；调小更省内存，但二阶近似可能变粗。",
            minimum=1,
        ),
        "line_search_fn": _parameter(
            "线搜索", "choice", "strong_wolfe",
            description="L-BFGS 使用的线搜索策略。",
            effect="当前只开放 strong_wolfe，不能随意改成未注册字符串；它通常更稳，但每步可能多次计算 closure。",
            choices=("strong_wolfe",),
        ),
    },
    ("time_window", "causal_window"): {
        "window_ends": _parameter(
            "时间窗终点", "number_list", [0.25, 0.5, 0.75, 1.0],
            description="按顺序扩展训练区域时使用的归一化时间终点。",
            effect="增加窗口数量会让因果推进更细但重采样阶段更多；减少窗口更简单。数值必须递增，最后通常为 1.0。",
        ),
    },
    ("transfer", "checkpoint_transfer"): {
        "checkpoint": _parameter(
            "检查点路径", "string", None,
            description="结构兼容的 OpenPINN checkpoint.pt 路径。",
            effect="它决定初始化来源，不是大小型超参数；换路径会换源模型，结构不兼容会在训练前报错。",
            required=True, placeholder="runs/source/checkpoint.pt",
        ),
        "trainable_last_linear_layers": _parameter(
            "继续训练的末端线性层数", "integer", None,
            description="留空表示继续训练全网络；填写后只训练最后若干线性层。",
            effect="留空适应能力最大但训练参数多；设为较小整数更快且保留更多源知识，但可能限制新问题拟合能力。",
            minimum=1,
        ),
    },
    ("domain_decomposition", "two_subdomain"): {
        "interface": _parameter(
            "子域界面位置", "number", 0.0,
            description="一维非均质 Poisson 两个材料子域的分界坐标。",
            effect="必须与材料系数真正跳变的位置一致；改错会让子网络划分和界面条件施加在错误位置，并非越大越好。",
        ),
    },
}


METHOD_SLOTS: dict[str, tuple[str, ...]] = {
    "baseline": ("baseline",),
    "modified_mlp": ("model_architecture",),
    "adaptive_activation": ("activation",),
    "fourier": ("feature",),
    "hard_boundary": ("constraint",),
    "adaptive_weights": ("loss_weighting",),
    "adaptive_sampling": ("sampling",),
    "adam_lbfgs": ("optimizer",),
    "time_window": ("sampling", "time_schedule"),
    "transfer": ("initialization",),
    "domain_decomposition": (
        "model_architecture",
        "feature",
        "constraint",
        "formulation",
    ),
}


METHOD_ORDER: dict[str, int] = {
    "baseline": 0,
    "domain_decomposition": 10,
    "modified_mlp": 20,
    "adaptive_activation": 30,
    "fourier": 40,
    "hard_boundary": 50,
    "transfer": 60,
    "adaptive_weights": 70,
    "adaptive_sampling": 80,
    "time_window": 80,
    "adam_lbfgs": 90,
}


def method_parameter_schema(method_name: str, variant: str) -> dict[str, dict[str, Any]]:
    """Return a defensive copy of the editable parameters for one variant."""

    try:
        return copy.deepcopy(METHOD_PARAMETER_SCHEMAS[(method_name, variant)])
    except KeyError as exc:
        raise ValueError(
            f"No parameter schema is registered for {method_name!r}/{variant!r}."
        ) from exc


def _validated_value(name: str, value: Any, schema: Mapping[str, Any]) -> Any:
    kind = schema["kind"]
    if value is None:
        if schema.get("required"):
            raise ValueError(f"Method option {name!r} is required.")
        return None
    if kind == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"Method option {name!r} must be boolean.")
        selected = value
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Method option {name!r} must be an integer.")
        selected = int(value)
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Method option {name!r} must be a number.")
        selected = float(value)
        if not math.isfinite(selected):
            raise ValueError(f"Method option {name!r} must be finite.")
    elif kind == "string":
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"Method option {name!r} must be a string or path.")
        selected = str(value)
        if schema.get("required") and not selected.strip():
            raise ValueError(f"Method option {name!r} cannot be empty.")
    elif kind == "choice":
        if value not in schema.get("choices", ()):
            raise ValueError(
                f"Method option {name!r} must be one of {schema.get('choices', [])!r}."
            )
        selected = value
    elif kind == "number_list":
        if not isinstance(value, (list, tuple)) or not value:
            raise TypeError(f"Method option {name!r} must be a non-empty number list.")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise TypeError(f"Method option {name!r} must contain only numbers.")
        selected = [float(item) for item in value]
    elif kind == "number_map":
        if not isinstance(value, Mapping):
            raise TypeError(f"Method option {name!r} must be a name-to-number mapping.")
        if any(
            not isinstance(key, str)
            or isinstance(item, bool)
            or not isinstance(item, (int, float))
            for key, item in value.items()
        ):
            raise TypeError(
                f"Method option {name!r} must map string names to numbers."
            )
        selected = {str(key): float(item) for key, item in value.items()}
    else:
        raise ValueError(f"Unknown parameter kind {kind!r} for {name!r}.")

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(selected, (int, float)) and not isinstance(selected, bool):
        if minimum is not None and selected < minimum:
            raise ValueError(f"Method option {name!r} must be >= {minimum}.")
        if maximum is not None and selected > maximum:
            raise ValueError(f"Method option {name!r} must be <= {maximum}.")
    return selected


def validate_method_options(
    method_name: str,
    variant: str,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize the explicitly supplied method options."""

    schema = method_parameter_schema(method_name, variant)
    selected = dict(options or {})
    unknown = sorted(set(selected).difference(schema))
    if unknown:
        available = ", ".join(schema) or "none"
        raise ValueError(
            f"Unknown option(s) for {method_name}/{variant}: {', '.join(unknown)}; "
            f"available options: {available}."
        )
    normalized = {
        name: _validated_value(name, value, schema[name])
        for name, value in selected.items()
        if value is not None
    }
    for name, item in schema.items():
        if item.get("required") and name not in normalized:
            raise ValueError(
                f"Method {method_name}/{variant} requires option {name!r}."
            )
    return normalized


def method(method_name: str, variant: str | None = None, **options: Any) -> MethodSelection:
    """Select one method variant with validated, editable hyperparameters."""

    from .method_registry import default_method_registry
    from .operator_protocol import default_operator_registry

    operator_registry = default_operator_registry()
    if operator_registry.contains(method_name):
        operator = operator_registry.resolve(method_name)
        if operator.implemented and operator.selectable:
            raise ValueError(
                f"Operator {operator.name!r} is not a PINN catalog method. "
                "solve() and build_config() stay on PINN families. "
                f"Select it with build_operator_config({operator.name!r}) and train_operator()."
            )

    spec = default_method_registry().resolve(method_name, variant=variant)
    normalized = validate_method_options(spec.name, spec.variant, options)
    return MethodSelection(spec.name, spec.variant, normalized)


def coerce_method_selection(value: MethodSelection | Mapping[str, Any]) -> MethodSelection:
    """Normalize a public method object or a small mapping."""

    if isinstance(value, MethodSelection):
        return method(value.method, value.variant, **dict(value.options))
    if not isinstance(value, Mapping):
        raise TypeError("Each methods item must be op.method(...) or a mapping.")
    selected = dict(value)
    method_name = selected.pop("method", selected.pop("name", None))
    variant = selected.pop("variant", None)
    nested_options = selected.pop("options", {})
    if selected:
        raise ValueError(
            "A methods mapping supports only method/name, variant and options; got: "
            + ", ".join(sorted(selected))
        )
    if not method_name:
        raise ValueError("A methods mapping requires method or name.")
    if not isinstance(nested_options, Mapping):
        raise TypeError("Method options must be a mapping.")
    return method(str(method_name), None if variant is None else str(variant), **dict(nested_options))


def validate_method_stack(
    selections: Sequence[MethodSelection],
) -> tuple[MethodSelection, ...]:
    """Reject duplicates and component-slot conflicts, then fix apply order."""

    if not selections:
        raise ValueError("methods must contain at least one selected method.")
    if len(selections) > 1 and any(item.method == "baseline" for item in selections):
        raise ValueError("baseline/plain_pinn cannot be combined with improvement methods.")

    occupied: dict[str, MethodSelection] = {}
    families: set[str] = set()
    for item in selections:
        if item.method in families:
            raise ValueError(
                f"Only one variant from method family {item.method!r} can be selected."
            )
        families.add(item.method)
        for slot in METHOD_SLOTS[item.method]:
            if slot in occupied:
                previous = occupied[slot]
                raise ValueError(
                    f"Methods {previous.method}/{previous.variant} and "
                    f"{item.method}/{item.variant} both control {slot!r} and "
                    "cannot be combined in the current framework."
                )
            occupied[slot] = item
    return tuple(sorted(selections, key=lambda item: METHOD_ORDER[item.method]))


__all__ = [
    "METHOD_ORDER",
    "METHOD_PARAMETER_SCHEMAS",
    "METHOD_SLOTS",
    "MethodSelection",
    "coerce_method_selection",
    "method",
    "method_parameter_schema",
    "validate_method_options",
    "validate_method_stack",
]
