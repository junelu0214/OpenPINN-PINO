:orphan:

算子配置器入门
==============

OpenPINN 的算子路径按 **模型 × 训练模式** 组合：模型可选 DeepONet 或 FNO，
``training_mode`` 可选 ``data_only``、``physics_only`` 或 ``hybrid``。
在 :doc:`method_configurator` 中设计方程和边界、选择组合，然后复制或下载可训练的
Python 脚本；训练在本地 Python 环境运行，不在浏览器中进行。

``data_only`` 与 ``hybrid`` 使用训练观测 CSV；可选的训后参考 CSV 只用于评估，
两者有各自的路径和列映射，不会互相替代。``physics_only`` 不需要训练观测，仍可
选择独立的训后参考数据。

需要逐单元运行训练、查看已提交输出时，从 :doc:`code_tutorials` 的二维 Poisson
FNO ``physics_only`` 主线开始；表单操作顺序见 :doc:`configurator_tutorial`。

进一步了解模型协议、残差路径与扩展点，请阅读 :doc:`developer_guide_operators`。
