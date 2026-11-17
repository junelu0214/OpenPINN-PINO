# 开发者指南重构与角色走查

日期：2026-10-14。作者的任务走查，不是真实外部用户研究。

## 原有问题与设计取舍

原首页图描述程序运行，而想贡献的人首先要回答的是：值不值得改公共代码、改哪里、怎样验证、怎样交付。
保留技术内容但调整层次；不改训练代码、不发布 GitHub、不建立 CI、不代选许可证。

- 首页改成贡献流程，包含三类任务分支、测试不通过返回、审阅修改返回、合并与发布责任。
- 六个子页面改为贡献任务分类；运行架构移到最后的参考页，函数附录不要求先读。
- 新增贡献计划、方法实施路线、修复路线、PR 与维护流程四节。
- 独立开发者 HTML 与站内页面保持相同分组顺序；保留既有锚点与技术细节。
- 使用者正文和 7 个 rst 文件的 SHA-256 与修改前逐一核对；未编辑使用者流程图分支。

## 路线一：我想贡献新的采样方法

入口：首页“新增算法 / 改进方法” → method-recipe → sampling.py 的扩展点 → 已有 residual_adaptive 接入样例 → method-done → testing → contribution-pr。

核对的实际来源：sampling.py、methods/adaptive_sampling/residual_adaptive.py、method_options.py、experiment.py 的 _residual_adaptive_sampler_factory，以及对应 methods 测试。
补充了组件算法与变体配置的区别、resample_every 的配置落点、白名单校验、slot/顺序、状态恢复和公平比较要求。
修订中移除了初稿里不存在的 activations.py / networks.py 路径，改为实际 features.py / models.py。
不提供一份未经执行的“通用新算法插件”冒充所有方法都能直接接入。

## 路线二：我只想修复孔洞编辑细节

入口：首页“修复缺陷 / 优化网页与文档” → maintenance-recipe → 现有 DXF 圆孔案例 → 函数契约 / 参数映射 → 针对性测试 → PR 模板。

区分纯样式修改和参数语义变化：前者不要求数值实验，后者必须检查生成脚本和后端。
补上最小输入、修复前失败回归、相邻孔不变、异常回退、生成文件不作为维护源等要求。
没有让小修复贡献者先阅读完整 Trainer 或全部函数索引。

## 路线三：维护者审阅与发布

入口：首页“提交可复核的 PR / 审阅通过？” → contribution-pr → 四项审阅问题 → 兼容/迁移及证据 → GitHub 发布准备。

明确 PR 合并与版本发布不是同一动作；新增方法精度主张不能由短跑背书。
走查当时本地未配置远程仓库与 CI。仓库软件现以 Apache License 2.0 授权，见根目录 `LICENSE` 与 `LICENSE_STATUS.md`。
文中将 Issue、fork、分支、PR、CI 等标为后续建议，不杜撰远程地址、维护者姓名或合并门禁。

## 验证范围

- 构建与手册标记示例验证；站内与离线目标、目录唯一覆盖、历史锚点和文档导航回归。
- developer_contribution_guide.test.cjs 验证任务入口、章节重排和运行架构下移。
- manual_diagrams.test.cjs 验证图中节点和返回箭头、原生链接及目标。
- 提取首页 SVG 用本地 Quick Look 渲染检查图示文字与箭头；这不是完整网页真实浏览器交互验收。
- 数值库没有实现变更；未做新算法性能实验、GitHub 实际提交或远程 CI 验证。
