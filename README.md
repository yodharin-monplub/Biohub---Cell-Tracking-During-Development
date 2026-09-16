# Biohub Cell Tracking competition workspace

This workspace is organized as a sequence of numbered, reproducible models.  A
new model is promoted only after it beats the current champion on an
embryo-held-out validation set or on a controlled Kaggle leaderboard probe.

Current status (2026-09-04):

| Model | Purpose | Status | Score evidence |
|---|---|---|---|
| `model1` | Freeze the strongest reproducible public dual-seed baseline | Submission `55969775` scored `0.934` public LB | Official visible-4 `0.895679`; visible 44b6 mean `0.934412`; exact reference hash |
| `model2` | Single-knob detector-recall probe | Rejected by official/local gates; not submitted | Official visible-4 `0.895576` (`-0.000103`) |
| `model3` | Single-knob bidirectional-linking probe | Candidate; validation required | None yet |
| `model4` | Single-knob division-precision probe | Candidate; validation required | None yet |
| `model5` | Synthetic division-geometry scorer | Research candidate; real calibration required | Held-out synthetic AUC `0.921` (not comparable to LB) |
| `model6` | Dense-synthetic detector-head fine-tune | v2 calibration complete; alpha-50 candidate gated | Matched-count synthetic top-count recall `0.8234` -> `0.8292`; real gate pending |
| `model7` | Confidence-aware FOCUS-3D real-image teacher | Complete; direct distillation rejected | Strong domain/count drift across the four movies |
| `model8` | Model6 alpha-50 head inside the frozen model1 pipeline | Complete; rejected, not submitted | Official visible-4 `0.897885`; `-0.035371` behind primary-only |
| `model9` | Checksum-pinned public “0.948” reproduction | Exact T4 reproduction; rejected, not submitted | Official visible-4 `0.895225`; exact public output hash |
| `model10` | Cross-embryo real division-geometry audit | Complete research audit | 151 real divisions; candidate AUC `0.9533`, poor graph calibration |
| `model11` | High-precision model5 division repair | Complete no-op; rejected | Threshold `0.82` added zero edges |
| `model12` | Exact secondary-only graph | Complete; rejected globally | Official visible-4 `0.907424` |
| `model13` | Primary/secondary whole-movie router | Complete; superseded | Official visible-4 `0.913911` |
| `model14` | Exact primary-only local-GPU audit | Complete; promoted to model15 | Official visible-4 `0.933257`; `+0.037577` over model1 |
| `model15` | Minimal primary-only code submission | Built and locally gated; explicit upload consent pending | Notebook SHA `f0185cb2…`; expected visible regression `0.933257` |
| `model16` | Primary detector-threshold sweep | Complete; global `0.965` retained | Seven strict-valid variants; best `0.933257` |
| `model17` | Native ILP division-cost sweep | Complete; rejected | 1–16 forks, zero division TP; topology missing |
| `model18` | Density-adaptive detector threshold | Complete provisional gain | Official visible-4 `0.933790`; count-only `+0.000534` |
| `model19` | Geometry-ranked division repair | Complete; rejected | 94 forks, zero TP; score `0.931985` |
| `model20` | Learned edge plus motion-cost sweep | Complete; promoted | Lambda `0.120`: score `0.939312`, raw edge Jaccard `0.934763` |
| `model21` | Fine motion-cost sweep | Visible-only champion; not promoted | Lambda `0.150`: visible-4 `0.939761`, but broad validation regresses |
| `model22` | 20-movie stratified motion validation | Complete; motion branch rejected | Lambda `0`: `0.892251`; `0.12`: `0.891489`; `0.15`: `0.890827` |
| `model23` | Source-guarded motion production ablation | Built at lambda zero; rejected for upload | Confirms model15 remains the safer production path |
| `model24` | Top-k alternative-parent association | GPU export prepared | Gives the ILP multiple parent choices instead of the current maximum of one |
| `model25` | ILP boundary-cost calibration | Complete; rejected globally | Best nonzero cost `0.125`: `0.892216` vs control `0.892251` |
| `model26` | Short-track component pruning | Complete; rejected | Length ≤4: `0.890497`; length ≤5: `0.890418` |
| `model27` | Selected-edge confidence filter | Complete; rejected | First cutoff `0.55` drops to `0.881114` |
| `model28` | Confidence plus trajectory smoothness | Complete; rejected | All 126 nontrivial rules lose; leave-one-movie-out delta `-0.000403` |
| `model29` | Spatial alternate-parent oracle | Complete; promising oracle | Top-2/radius-7 exposes 372 true edges; `0.959824` optimistic ceiling, not a score |
| `model30` | Conservative gap closing | Complete; promoted | Exact broad `0.897103` vs `0.892251`; LOMO stable and both families improve |
| `model31` | Primary + internal gap closing | Production notebook ready locally | Deterministic and exact-parity gates pass; upload requires consent |
| `model32` | Gap tracklet-length filter | Complete; rejected | None of 200 stricter rules beats model30/model31 |
| `model33` | Motion-aware endpoint matching | Complete; rejected | LOMO loses `0.000320`; 6bba regresses and top-k adds no TP |
| `model34` | Broad division-repair retest | Promising; high variance | `0.899421`; 1/5 division TP but visible-four had zero TP |
| `model35` | Gap closing + division repair | Promising; high variance | Broad `0.903186`, but visible-four regresses to `0.932675` with zero division TP |
| `model36` | Real-label division ranker | Grouped validation complete | OOF AUC `0.973221`; 75.4% precision and 35.9% recall at frozen fold operating points |
| `model37` | Gap closing + real-label division repair | Broad numerical champion; high variance | Broad `0.905823` with 2 division TP; visible-four `0.932706` with zero TP |
| `model38` | Precision-focused real-label division repair | Current broad numerical champion | Broad `0.907439` with 2 TP / 14 FP; visible-four remains `0.932706` |
| `model39` | 90%-precision real-label division repair | Complete; rejected | Broad `0.896989`, zero division TP; visible-four inert |
| `model40` | 95%-precision real-label division repair | Complete; inert control | Broad `0.897109`, zero division TP; visible-four inert |
| `model41` | Detector-domain division audit | Complete | Three direct positives in two 6bba movies; no basis yet for a transferable learned selector |
| `model42` | Sparse low-confidence division rescue | Train-fit numerical champion | Broad `0.912769` with 3 TP / 14 FP; visible-four byte-identical to model38 |
| `model43` | Production model42 branch | Notebook ready locally | Deterministic `e5842952…`; exact broad/visible parity and static gates pass |
| `model44` | Conservative parent replacement search | Complete; cautious candidate | Fixed rule `+0.000515` and both families improve; adaptive LOMO is negative |
| `model45` | Parent replacement + gap closing | Rejected by independent visible gate | Broad `0.897617`, but visible `0.930437` versus model31 `0.933944` |
| `model46` | Parent replacement + division branch | Stopped at prerequisite gate | Model45's visible regression makes this high-variance composition unsuitable |
| `model47` | Grouped learned gap ranker | Rejected by leave-one-movie-out gate | LOMO `0.895078`, `-0.002029` versus reconstructed model30 |
| `model48` | Top-five alternative-parent ILP | Promoted to production branch | Broad `0.904193` at p=0.40, `+0.011937`; both families improve |
| `model49` | Broad density-adaptive detector check | GPU-blocked | Local GPU is occupied; no scored output exists |
| `model50` | Top-five division reachability audit | Complete | All 5 broad real divisions have both daughter candidates |
| `model51` | Top-five association + gap close | Promoted to model56 | Broad `0.904520`, `+0.000327`; gap selected 18/20 LOO folds |
| `model52` | Native top-five division-cost sweep | Rejected structural pilot | Cost 0.70 creates 186 forks in first 44b6 graph |
| `model53` | Global ranked division repair on top-five graph | High-variance rejection | Broad `0.908259`, but 1,105 forks and 19 scored division FP |
| `model54` | Top-five/gap + sparse division rescue | High-upside branch | Broad `0.924587`, but division gain is one-movie concentrated |
| `model55` | Sparse-rescue ablation without gap close | Complete ablation | Broad `0.924260`; model54 retains the small gap gain |
| `model56` | Production top-five + conservative gap notebook | Rejected by visible gate | Broad `0.904520`, but fresh visible-four score `0.929396` trails primary-only `0.933257` |
| `model57` | Production top-five/gap + sparse rescue notebook | Rejected by visible gate | Sparse gate is a visible-four no-op; inherits model56 `0.929396` |
| `model58` | p=0.35 top-five floor interpolation | Rejected | Broad `0.902697`, `-0.001497` vs p=0.40; loses both families |
| `model59` | Top-five ILP with small motion prior | Cautious candidate | Broad `0.904599`, `+0.000406` vs p=0.40; 17 wins / 3 losses but 44b6 regresses |
| `model60` | Motion-aware top-five + conservative gap close | Rejected by visible gate | Broad `0.904986`; fresh visible-four `0.927826`, below model56 and primary-only |
| `model61` | Model60 + frozen sparse division rescue | Rejected by visible gate | Broad division gain is one-movie concentrated; visible rescue is a no-op |
| `model62` | Motion-prior stability sweep | Rejected by runtime gate | Lowest planned weight did not complete its first large exact ILP graph |
| `model63` | Higher motion-prior probes | Rejected by runtime gate | The next stronger penalty also failed to complete the first large exact ILP graph |
| model64 | Strict-LOO top-five edge-calibration audit | Complete | Candidate AUC 0.994174 -> 0.994597; top-1 choices 13,537 -> 13,543 |
| model65 | Strict-LOO calibrated top-five association | Cautious OOF branch | Strict OOF 0.904767, but only 6/20 movie gains |
| model66 | Strict-LOO calibrated association + frozen gaps | Best strict OOF; cautious | Strict OOF 0.905223, +0.001029 vs p=0.40; uneven movie profile |
| model67 | Half-blend calibrated association + gaps | Rejected | Strict OOF 0.903904, below model66 by 0.001319 |
| model68 | Rank-preserving calibrated association + gaps | Near-tie rejected | Strict OOF 0.90522234, 0.00000052 below model66 |
| model69 | Within-family calibrated association + gaps | Rejected | Strict OOF 0.904879, below model66 by 0.000344 |
| model70 | Calibrated parent-replacement/router audit | Complete | Isolated replacements harmful; nested family router `0.904938` |
| model71 | Family-routed calibration/motion composition | Tuned broad candidate | Exact broad `0.905528`; above model66 numerically but not strict nested OOF |
| model72 | Target-conditional pairwise ranker audit | Complete | Strict-LOO top-1 improves from 13,543 to 13,549; graph solve promoted to model73 |
| model73 | Rank-preserving pairwise strict-LOO association | Rejected | Strict OOF `0.904435`, `-0.000788` vs model66; 15 wins but large weighted losses |
| model74 | RTX 4050 visible-four top-five validation | Complete | Model56 `0.929396`; model60 `0.927826`; both below primary-only `0.933257` |
| model75 | Visible raw+gap sparse-division gate | Rejected no-op | 14,283 candidates scored, zero accepted; inherits `0.929396` |
| model76 | Visible motion+gap sparse-division gate | Rejected no-op | 14,168 candidates scored, zero accepted; inherits `0.927826` |

Model21 improves the frozen public control by `0.044082` on the exact
visible-four scorer, but the gain does not generalize: on model22's 20-movie
stratified set, lambda `0.12` and `0.15` regress the official weighted score by
`0.000761` and `0.001424`. The family router also fails leave-one-out testing.
The global lambda-zero primary model therefore remains the production control.
Parameter-only variants are never described as gains until a score receipt
exists.

## Bootstrap

The project environment contains the official `kaggle` and `kagglehub`
clients.  Authenticate once, accept the competition rules, and download the
data:

```bash
export KAGGLE_CONFIG_DIR="$PWD/.kaggle"
.venv/bin/kaggle auth login --no-launch-browser
.venv/bin/python scripts/download_competition.py --output-dir data/raw
```

The competition archive is about 87.6 GB.  Keep at least 100 GB free.

Generate the frozen public notebook and its controlled variants from the
downloaded Kaggle API payload:

```bash
.venv/bin/python scripts/bootstrap_models.py \
  --kernel-payload /tmp/evgen_kernel \
  --reference-submission /tmp/evgen_submission.csv
```

Validate any produced CSV before spending a Kaggle submission:

```bash
.venv/bin/python scripts/validate_submission.py \
  model1/reference_submission.csv --test-dir data/raw/test
```

Once the training data are extracted, reproduce the actual organizer metric
against any overlapping train/visible-test IDs (not the notebook's simplified
proxy):

```bash
.venv-gpu/bin/python scripts/score_submission.py \
  model1/reference_submission.csv \
  --train-dir data/raw/train \
  --json-out model1/official_visible_score.json
```

For the frozen control this yields `0.8956793462` across all four visible
movies; the two-44b6 mean is `0.9344116378`, matching its displayed public LB.
Treat this as a scorer calibration and a regression test, not an unbiased final
rerun estimate: robust promotion still requires leave-one-embryo-out models.

Each `modelN/readme.txt` records the exact hypothesis, changes, evidence, and
promotion gate for that model.

Run every GPU notebook with an explicit T4 request:

```bash
.venv/bin/kaggle kernels push -p model1 --accelerator NvidiaTeslaT4
```

This is required with Kaggle's current image: its bundled PyTorch no longer
contains kernels for the legacy P100 (`sm_60`).  A plain `enable_gpu: true`
request may still schedule a P100 and fail before inference begins.  The
experiment ledger in `experiments.csv` records both failed infrastructure runs
and scored runs so they are never confused with model regressions.

## Competition facts that drive the design

- Score: adjusted edge Jaccard + `0.1 * division Jaccard`.
- Node matching uses physical distance with a 7 micrometer cutoff.
- Voxel scale `(z, y, x)` is `(1.625, 0.40625, 0.40625)` micrometers.
- Edges must connect strictly consecutive frames.
- The hidden test rerun is roughly the size of train and must finish within 12
  hours with internet disabled.
- This is a code competition: Kaggle must rerun a notebook that writes
  `/kaggle/working/submission.csv`; uploading a CSV directly is not sufficient.

## Provenance and fair-play policy

`model1` is an unchanged snapshot of the Apache-2.0 public Kaggle notebook
`evgendvorkin/biohub-0-934-lb-proxy-score-0-9384`, version 27.  Its required
model datasets are public CC0 assets.  The patched metric exploit is explicitly
out of scope.  Every subsequent change must preserve graph validity and be
tested against the current official metric implementation.
