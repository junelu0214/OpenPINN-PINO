# Third-party notices

OpenPINN's visual problem-designer preview redistributes the following pinned,
unmodified browser assets. OpenPINN's own integration code is maintained
separately in `docs/_static/visual_problem_designer.js` and `.css`.

## Blockly 13.3.0

- Project: https://github.com/RaspberryPiFoundation/blockly
- Package: https://www.npmjs.com/package/blockly/v/13.3.0
- Purpose: local visual block workspace for PDE expressions.
- License: Apache License 2.0.
- Local license copy: `docs/_static/vendor/blockly/LICENSE`.
- Redistributed files: `blockly_compressed.js`, `msg/zh-hans.js`, and the
  runtime `media/` assets from the published npm package.

## KaTeX 0.18.7

- Project: https://github.com/KaTeX/KaTeX
- Package: https://www.npmjs.com/package/katex/v/0.18.7
- Purpose: local mathematical formula rendering.
- License: MIT.
- Local license copy: `docs/_static/vendor/katex/LICENSE`.
- Redistributed files: `katex.min.js`, `katex.min.css`, and the fonts referenced
  by that stylesheet from the published npm package.

No source code from external PINN implementations is included by this feature.

## jsdom 30.1.2 (development tests only)

- Pinned in `package.json`; exact dependency tree is recorded in `package-lock.json`.
- Purpose: local DOM regression tests for the configurator and manuals.
- License: MIT; installed license text is in `node_modules/jsdom/LICENSE.txt`.
- Not included in the standalone manuals or browser runtime assets.
- `node_modules/` is a local installation, not a source or delivery directory.
  If redistributing installed dependencies, retain each dependency's own license
  notices; this entry does not replace transitive dependency licenses.
