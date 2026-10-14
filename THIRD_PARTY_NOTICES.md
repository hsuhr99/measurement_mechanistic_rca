# Third-Party Notices

This repository includes original code for measurement-mechanistic RCA and a bundled replication package. Original code developed for this repository is licensed under CC BY 4.0; see `LICENSE`.

Bundled or adapted external code remains under the licenses of the respective upstream projects. This file summarizes the external components currently included in the repository.

| Component | Bundled path | Upstream/source | License status |
| --- | --- | --- | --- |
| dodiscover, including NoGAM/CAM/SCORE-style top-order discovery code | `replication_neurips/src/competitors/dodiscover/` | https://github.com/py-why/dodiscover | MIT. License text is bundled at `replication_neurips/src/competitors/dodiscover/LICENSE`. |
| TOPIC | `replication_neurips/src/competitors/topic/` | https://github.com/srhmm/topic/ | MIT. License text is bundled at `replication_neurips/src/competitors/topic/LICENSE`. |
| TOPIC embedding helper | `replication_neurips/src/competitors/topic/src/topic/util/embedding.py` | Embedded in the TOPIC codebase | MIT notice included in the source file: Copyright (c) 2018 Mojmir Mutny, ETH Zurich. |
| TOPIC Gaussian-process helper code | `replication_neurips/src/competitors/topic/src/topic/scoring/models/mdl_gp.py`, `replication_neurips/src/competitors/topic/src/topic/scoring/models/mdl_fourierf.py` | Derived from scikit-learn-style Gaussian process code | BSD 3-Clause notice indicated in the source files. |
| NOTEARS baseline | `replication_neurips/src/competitors/topic/src/baselines/notears/` | https://github.com/xunzheng/notears | Apache License 2.0. License text is bundled at `replication_neurips/src/competitors/topic/src/baselines/notears/LICENSE`. |
| SCORE baseline | `replication_neurips/src/competitors/topic/src/baselines/score/` | Upstream SCORE baseline used by the replication package | GNU AGPL v3. License text is bundled at `replication_neurips/src/competitors/topic/src/baselines/score/LICENSE`. Treat this component carefully when redistributing or packaging. |
| M2OE baseline | `replication_neurips/src/competitors/M2OE/` | https://github.com/AIDALab-DIMES/M2OE | MIT. License text is bundled at `replication_neurips/src/competitors/M2OE/LICENSE`. |

Before a public release, verify each bundled component against the exact upstream revision used in the replication package and keep the upstream license files with any redistributed copies.
