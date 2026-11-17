# License status

OpenPINN is licensed under the Apache License, Version 2.0.
The full license text is [`LICENSE`](LICENSE) at the repository root.
The copyright notice in that file is `Copyright 2026 Jun Lu`.

This repository-wide license covers OpenPINN's own source.
It does not relicense third-party assets, and it does not change the
license boundaries recorded for reference papers and external repositories.
Method documentation records papers, repository links, and known license
boundaries. Papers explain algorithms but do not grant a software license.
Third-party browser assets are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## 2026-08-29 method-expansion audit

The following paper repositories report no detected license in GitHub metadata
and expose no top-level `LICENSE` file. OpenPINN therefore treats them as
license-unclear and did not copy, translate, or adapt their source code:

- https://github.com/hsbhc/AMAW-PINN
- https://github.com/rbischof/relative_balancing
- https://github.com/wangyong1301108/MMPINN
- https://github.com/levimcclenny/SA-PINNs

The new implementations use equations and algorithm descriptions from the
papers only. The Nabian importance-sampling and Kendall uncertainty-weighting
variants likewise use paper equations; no external implementation was copied.
