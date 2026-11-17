# 物理参数目录与数据贡献

运行时内置目录位于 `src/openpinn/physics_catalog.json`，校验与残差解释器位于 `src/openpinn/physics.py`。本目录保存贡献约定和验收计划，不是第二套运行时数据源。

## 数据层次

| 数据 | 作用 | 不应表达的含义 |
| --- | --- | --- |
| 参数定义 `parameters` | 记录 id、分类、显示名、符号、单位、说明和可选值 | 仅定义一个“扩散系数”不代表其值已经适用于某种岩土或温度 |
| 材料记录 `materials` | 记录具体材料/工况采用的参数值 | 演示数值不应标成实测数据或权威推荐值 |
| 方程模板 `templates` | 提供可编辑的单场方程、参数与单位组合 | 模板不是完整的几何、边界、初值或训练工程 |

学习率、网络结构、训练步数和方法变体继续由 `method_options` 与训练配置管理。向物理参数库提交数据，不会增加训练方法。

## 格式与约束

内置目录的 `schema_version` 为 `1`。目录含显式 `units`、`categories`、`parameters`、`materials` 和 `templates`。

单位通过 `factor` 与 `dimensions: [M, L, T, Theta]` 声明到 SI 的乘法换算和量纲。二维 x/y 使用同一单位。新增单位必须同时评估 Python 和浏览器校验、求导换算、显示与往返导出；浏览器参数包不接受自行定义单位。

不得在相同 schema 版本下改变已发布单位的换算含义（例如 year 的秒数）；这会改变旧方程快照的计算结果。此类不兼容变更须提升规格版本并提供迁移规则。

参数定义使用稳定 `id`，并给出 `category`、`label`、`symbol`、`unit`、`description`，定义的 `value` 必须为 `null`。具体材料值单独放入材料记录，不应默认为零或填入看似可信的材料常数。同一个方程内公式符号需唯一，并避开未知场、坐标和保留函数名；目录中不同定义可有相同建议符号。参数值为有限常数；空间变系数通过公式中的坐标关系表达。

模板包括可读 `expression_text` 和包含安全 AST 的 `spec`。两者必须描述同一公式。`spec` 至少声明：

```text
schema_version: 1
dimension: 1 或 2
kind: static 或 dynamic
field: 单个未知场名称
field_unit: 内置单位 id
coordinate_units: x / y / t 的单位配置
expression: 安全 AST
parameters: 完整参数记录列表
```

`expression_source` 可以保存公式原文字。模板中的初值和边界值为建议说明；应用模板不能悄悄覆盖配置器已有的几何或约束。示范模板须明确 `is_demo: true`，示例参数值不能作为实验测定值引用。

离线扩展包只允许 `schema_version`、`parameters`、`materials`，不允许注入 `units`、`templates` 或代码。扩展包和完整内置目录不是同一种文件。导入须先完整校验再提交状态；任一记录出错时不得留下部分导入的数据。

可复制 `examples/physics/site_diffusion_parameter_pack.json` 制作离线包。材料 `parameters` 用 `{id, value, unit}` 引用定义。相同定义重复导入可复用，冲突定义与重复材料 id 拒绝。`validate_physics_catalog` 针对带分类和模板字段的完整目录；浏览器包用 `OpenPINNPhysicsEditor.validateParameterPack` 校验。新增物理单位或函数属于代码贡献，不是导入数据文件即可开放的能力。

## 贡献流程

1. 在 Pull Request 中提交数据修改，并说明参数/材料/模板属于哪一层。新增材料值同时提供来源、材料/工况、测量或引用条件、单位和可共享依据；没有可靠来源时明确标为演示值。
2. 用 `validate_physics_catalog` 检查完整目录，用 `validate_physics_spec` 检查受影响的模板。离线参数包另需通过浏览器导入校验，不能仅以 JSON 能解析作为通过标准。
3. 运行相关 Python 与 JavaScript 测试，补充能独立发现单位、量纲或表达式错误的例子。修改单位换算时至少比较一种非 SI 表达与同物理量的 SI 表达。
4. 运行 `python scripts/build_docs.py`，检查说明、生成代码和离线导入/导出行为，再由维护者合并。

当前页面没有接收社区数据的服务器。导入只影响本地编辑状态，导出只生成本地 JSON；不会自动向仓库上传，也不会替代 Pull Request 审查。

## 迁移与适用范围

生成代码嵌入完整 `spec`，无需原始参数包文件；运行电脑仍需安装包含物理解释器的兼容 OpenPINN 版本。方程工程导出仅包含方程和物理参数，不是完整训练项目快照。

当前安全入口支持一维/二维单个标量场，静态或一阶时间问题，嵌套求导总阶不超过二阶。它不提供二阶时间导数、多场 Biot 耦合、材料分区界面或物理通量边界的自动建模。通过校验不证明物理模型正确、数据适用或训练收敛。

详细验收标准见 [ACCEPTANCE.md](ACCEPTANCE.md)，用户说明源文件见 `docs/physics_library.rst`。验收计划不等于已通过的报告。
