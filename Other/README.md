# Biohub Cell Tracking During Development: 206 experiments, one honest lesson

Our full working record for the Kaggle competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(September 2026). Every experiment we ran is here as a numbered folder with its code, a `readme.txt` explaining what
it tested and why, and a `score.txt` with the result. Most of it failed, and the failures are documented as
carefully as the successes, because that is where the lessons are.

| | |
|---|---|
| **Task** | Track every cell nucleus through 3D light-sheet time-lapse movies of developing embryos: detect the cells in each frame, link them across frames, and find cell divisions. |
| **Metric** | adjusted edge Jaccard + 0.1 × division Jaccard ([official definition](Other/vendor/official/metrics.md)); the adjustment penalises predicting more cells than the organisers' estimate. |
| **Best public leaderboard** | **0.947** (`Model/model167`) |
| **Final private leaderboard** | *to be added when revealed* |
| **Final submissions** | `model167` (public 0.947) and `model209` (public 0.945) |

## The short version

1. **Our best submission is a verified reproduction of a public notebook.** `model167` re-runs the public
   [zhincez 0.947 notebook](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets)
   exactly (the notebook is pinned by SHA-256, and only file paths were changed). Nothing we built on top of it scored
   higher on the public leaderboard.
2. **Local scores on the training movies lie.** The public checkpoints were trained on those same movies, so the
   pipeline scores above 0.96 on them. On an embryo it has never seen, it scores about 0.80. The training data
   come from only two embryos (`44b6`, `6bba`), so the only honest local test trains on one embryo and scores on
   the other. `Other/VALIDATION_AUDIT.txt` documents how this was found; `Model/model182` is the resulting testbed.
3. **On that honest testbed, the real levers were training choices, not post-processing.**
   Tuning the post-processing thresholds moved the honest score by less than ±0.005. A second independent seed gave +0.0095
   (`model186`), and roughly doubling the training budget gave +0.0196 (`model189`). A third seed added nothing
   (`model187`), and synthetic pretraining hurt (−0.037, `model184`).
4. **But swapping those better-trained weights into the public pipeline lost on the leaderboard every time.**
   All five such submissions scored 0.900–0.942 against 0.947 (`model190`–`model196`). The pipeline's thresholds are
   calibrated around the public weights. Comparing predicted node counts on the 4 visible test movies against the
   public reference predicted the leaderboard order exactly, at no Kaggle GPU cost (`model199`).
5. **Divisions are rare and noisy, and public claims about them were not verified.** There are only 151 annotated
   divisions in 199 training movies. A public division detector claiming AUC 0.845 scored 0.562 when tested
   (`model206`). Our own detector reached AUC 0.907 (`model207`). Adding its divisions gained +0.0018 under the
   official metric on 40 training movies (`model208`), but the submission scored 0.945 on the public
   leaderboard (`model209`).
6. **Score with the official code, not a notebook's approximation.** The notebook's built-in scorer only
   approximates the organisers' division rules, and small division changes are exactly where that matters. The official scorer is vendored in `Other/vendor/official`
   and wrapped by `Other/scripts/score_submission.py`.

## Leaderboard submissions

| Model | What changed vs the public pipeline | Public LB |
|---|---|---|
| [model1](Model/model1/readme.txt) | frozen reproduction of an earlier public 0.934 notebook (starting point) | 0.934 |
| [model167](Model/model167/readme.txt) | exact reproduction of the public 0.947 notebook | **0.947** |
| [model174](Model/model174/readme.txt) | + conditional division guard | 0.945 |
| [model181](Model/model181/readme.txt) | motion-relink stage switched off | 0.945 |
| [model190](Model/model190/readme.txt) | our long-trained checkpoint replacing one / both public checkpoints | 0.920 / 0.900 |
| [model193](Model/model193/readme.txt) | public primary checkpoint fine-tuned (lr 2e-5, 24 epochs) | 0.938 |
| [model195](Model/model195/readme.txt) | both public checkpoints fine-tuned | 0.940 |
| [model196](Model/model196/readme.txt) | gentle fine-tune (lr 1e-5, 12 epochs) | 0.942 |
| [model209](Model/model209/readme.txt) | + extra divisions proposed by our own detector (model207) | 0.945 |

`Model/README.md` lists all 206 model folders with a one-line summary each.

## The pipeline (public 0.947 notebook)

Two independently trained TemporalUNet3D + node-transformer networks (the primary and secondary "seeds") detect
nuclei and score candidate links between detections in consecutive frames, with 8-view test-time augmentation. A
separate 3D U-Net centre-prior model (DeepCenter) supplies an extra confidence signal that the gap-closing and
division checks use. An integer linear program (ILP) selects the
tracks. Post-processing then: filters and re-links edges by motion, closes one- and two-frame gaps, adds "safe"
divisions, filters division geometry, and removes short tracks. It runs as a Kaggle code submission on a T4 GPU
with no internet. The checkpoints are public Kaggle datasets by pilkwang (see Credits).

## Repository layout

```
AGENTS.md            the instructions the project was run under (see "How this was made")
Model/README.md      index: one line per experiment
Model/modelN/        one experiment: source code, readme.txt (always), score.txt (when scored),
                     weights/ (checkpoints we trained, when small enough), kaggle/ (submitted notebooks + outputs)
Other/REPRODUCE.md   how to rebuild the environment and rerun the key experiments
Other/scripts/       shared tools: official scorer wrapper, submission validator, Kaggle helpers, backup
Other/vendor/official/  the organisers' metric code (BSD-3-Clause)
Other/VALIDATION_AUDIT.txt  how the train-movie scores were shown to be misleading
Other/experiments.csv       log of Kaggle kernel runs and submissions (up to 2026-09-21; later ones are in score.txt files)
Other/WORKSPACE_NOTES.md    the running status table kept during the first phase of the project
```

Not included: the competition data (download it from Kaggle), the public checkpoints (Kaggle datasets, see
Credits), virtual environments, credentials, and files over 45 MB.

## Reproducing

See [`Other/REPRODUCE.md`](Other/REPRODUCE.md). In brief:

1. Download the competition data into `Data/competition` and the three pilkwang checkpoint datasets into
   `Data/public_checkpoints`.
2. Create the two Python 3.11 environments described there: `Other/.venv` for tools, and `Other/.venv-gpu`
   for CUDA and tracking.
3. Reproduce the 0.947 submission locally with `Model/model167/run_reproduction.ps1`, then score any
   submission CSV on training movies with
   `Other/.venv-gpu/Scripts/python.exe Other/scripts/score_submission.py <csv> --train-dir Data/competition/train`.

Things to know when reading the code:
- **Models 1–166 were run on Linux** and their readmes use Linux paths. Their shell scripts are the record,
  not a promise that they run unchanged.
- **Models 167+ were run on a Windows laptop** with an RTX 4050 (6 GB). Bulk working files lived in
  `C:\biohub_data\work\<model>`, because Windows' 260-character path limit breaks the pipeline's temporary
  files inside a deep project path.
- **Rented GPUs:** a few experiments trained on rented Vast.ai / RunPod GPUs (for example `model91`, `model104`,
  `model166`, `model185`). Those machines no longer exist.

## How this was made

The project was run by an AI coding agent (Claude Code), working under the instructions in `AGENTS.md` and
steered by the repository owner. Each experiment's `readme.txt` was written at the time, including the
hypotheses that turned out wrong. Later results, such as leaderboard scores, were added as dated notes rather than
rewriting the original text.

## Credits

- Public notebooks reproduced and studied (`Model/model1`, `model9`, `model167`, `model178`, `model179`):
  [zhincez](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets),
  [karl0106](https://www.kaggle.com/code/karl0106/biohub-p26-0947-fast-repro) and
  [busyaprime](https://www.kaggle.com/code/busyaprime/biohub-0-942-lb-one-knob-past-the-public-line).
  Copies are kept for provenance and remain under their authors' terms (Kaggle notebooks default to Apache 2.0).
- Public checkpoints: the pilkwang Kaggle datasets `biohub-tracking-support-pack-50ep-v1`,
  `biohub-temporal-unet3d-seed314159-v1` and `biohub-deepcenter-unet3d-center-prior-v1`.
  Checkpoints under `Model/*/weights` that were fine-tuned from them derive from those weights.
- Official metric code: [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition),
  BSD-3-Clause, © 2026 Thibaut Goldsborough, vendored unchanged in `Other/vendor/official` with its licence.
- Synthetic data generator (`Model/model184`): José Freitas's CC0 generator (Kaggle kernel
  `josefreitasalvesneto/biohub-synthetic-dataset`).
- DivNet artifact tested in `Model/model206`: Kaggle dataset `giorgosi/biohub-divnet-v2`.
- Competition data and task: CZ Biohub and the competition organisers. The data is not redistributed here.
