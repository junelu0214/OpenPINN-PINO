修复功能与改进网页
================================================================

从一个可复现的小问题开始，定位源文件，写回归，再修复并验证。
纯文案与布局调整无需牵动数值算法；改变参数含义时必须验证生成代码和后端行为。

先按 :doc:`developer_guide_start` 准备环境；数学接口见 :doc:`developer_guide_modeling`，
方法组装见 :doc:`developer_guide_runtime`。完成改动后按 :doc:`developer_guide_delivery` 验收，
函数位置可在 :doc:`developer_guide_reference` 查询。

物理参数库的数据契约与维护入口见 :doc:`physics_library` 及仓库
``catalog/README.md``。它独立于训练方法参数注册，不要混用两个配置层。

The configurator JSON payload also carries ``operator_hooks`` for the
coordinate/grid operator protocol. The operator path selects a canonical
model (DeepONet or FNO) and a separate ``training_mode``
(``data_only`` / ``physics_only`` / ``hybrid``). PINO is not a model option.
PINN checkboxes still come from ``families``. See
:doc:`developer_guide_operators`.

The same page has a solution-path control. ``operator`` reuses the existing
one- or two-dimensional geometry, PDE, boundary, and visualization controls and
writes the resulting ``PDEProblem`` into a local ``train_operator_problem`` script.
DeepONet uses coordinate autodiff and FNO uses structured-grid derivatives on
that exact equation. data_only/hybrid require a local observation CSV that is
independent of the optional post-train reference CSV; neither file is
uploaded, and one side is never reused as the other. Copy and download stay disabled for an unsupported dimension, an illegal hook option,
or a non-selectable operator. A collapsed advanced section is populated from
the derivative, sampler, and optimizer component registries; its selections
are written to ``EXTENSIONS`` in the local script. The browser never trains
or solves.

The plate-with-hole codegen and tests remain, but the plate is a tutorial at
:doc:`plate_with_hole_demo`, not a configurator solution path. 3D, contact,
and browser-side training are outside this wiring.

.. openpinn-manual:: developer configurator

