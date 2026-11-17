验证、提交 PR 与版本维护
================================================================

贡献者提交可复核证据与迁移说明；维护者审阅、决定合并，并独立安排版本发布。
本页的 GitHub 流程是建议约定，不意味着已经启用 CI、分支保护或自动发布。

双语文档维护
------------

中文 ``.rst`` 与 notebooks 是源文档；英文 gettext 目录位于
``docs/locales/en/LC_MESSAGES/``。修改可见标题或已翻译教程正文后，先运行
``python -m sphinx -b gettext docs /tmp/openpinn-gettext`` 对照更新对应 ``.po``，
再运行 ``python scripts/build_docs.py``。后者以 ``-W`` 分别构建默认中文站点
``docs/_build/html/`` 与英文站点 ``docs/_build/html/en/``，且不会执行 notebooks。
新增可见侧栏页时，至少应翻译标题；教程主线应同时翻译正文。每页语言切换链接由
``docs/conf.py`` 与 ``docs/_templates/layout.html`` 生成。

.. openpinn-manual:: developer delivery

