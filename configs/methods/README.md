# 方法家族与具体变体

每个一级文件夹是一类方法；文件夹中的 YAML 是该家族下的具体实现。
`method.name` 选择家族，`method.variant` 选择具体实现。省略 `variant` 时使用
该家族当前的默认变体，以兼容旧调用。

自适应采样目前有 ``residual_adaptive``、``residual_movement``、
``gradient_movement``、``importance_sampling``；损失平衡目前有
``self_adaptive``、``uncertainty_weighting``、``relobralo``、
``magnitude_power``、``pointwise_self_adaptive``。其他家族仍保留第一种实现。

继续增加二级方法时，必须同时增加源码、配置、小测试、方法说明和对照结果，
不能只添加一个没有算法依据的名字。
