# How to reproduce this project without the Claude session

Layout (AGENTS.md, re-laid out 2026-09-20):

```
AGENTS.md                      only file at the root
Data\competition               Kaggle competition data (junction -> C:\biohub_data\raw, 88 GB, kept outside OneDrive)
Data\public_checkpoints        public pilkwang checkpoints + HOCT package (junction -> C:\biohub_data\public_models)
Data\public_notebooks          pulled public Kaggle notebooks used for study (junction -> C:\biohub_data\public)
Data\synthetic_generated       locally generated CC0 synthetic sequences (junction -> C:\biohub_data\work\model184\synthetic)
Data\gt_division_inventory.csv annotated divisions per train movie
Model\modelN                   one folder per experiment: source, readme.txt, score.txt when scored, weights\ when trained
Other\                         .env (API keys), email.txt, .venv (CPU), .venv-gpu (CUDA), scripts\, tests\, README.md,
                               VALIDATION_AUDIT.txt, experiments.csv, this file
```

Bulk working files (prediction graphs, logs, run directories) live in `C:\biohub_data\work\...` because
zarr/geff temp paths inside the OneDrive project exceed the 260-character Windows limit.

Scripts inside `Model\modelN` compute `ROOT = <project>\Model`, so `ROOT / "model167/..."` and
`from model182...` imports keep working; data, venv, scripts and `.env` are reached through `ROOT.parent`.
Run PowerShell scripts from anywhere; they `Set-Location` to `Model\` themselves.
Models 1-166 were written on Linux before the move to this laptop; their readmes are the record, their
shell scripts are not expected to run on Windows.

## 0. Environment (once)
```
winget install astral-sh.uv
uv venv Other\.venv --python 3.11 ;  uv pip install --python Other\.venv\Scripts\python.exe kaggle kagglehub numpy pandas
uv venv Other\.venv-gpu --python 3.11
uv pip install --python Other\.venv-gpu\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python Other\.venv-gpu\Scripts\python.exe tracksdata "zarr>=3.0.10,<4" pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig google-crc32c numcodecs scikit-image scipy numba networkx pandas kaggle kagglehub ipython joblib tqdm matplotlib "hoct[bioio]"
```
Kaggle auth: `$env:KAGGLE_API_TOKEN = <KAGGLE_API_KEY from Other\.env>`.
Data: `powershell -File Other\scripts\run_download.ps1` (88 GB). Public checkpoints: Kaggle datasets
`pilkwang/biohub-deepcenter-unet3d-center-prior-v1` -> `Data\public_checkpoints\deepcenter`,
`pilkwang/biohub-temporal-unet3d-seed314159-v1` -> `...\secondary-seed`,
`pilkwang/biohub-tracking-support-pack-50ep-v1` -> `...\support-pack`.

## 1. Best submission so far: model167 (public LB 0.947)
Exact public notebook `Model\model167\public\zhincez\...ipynb` (SHA-256 38bca69a...).
- Kaggle: `Model\model167\kaggle\` (kernel-metadata.json + submission.ipynb); push with
  `kaggle kernels push -p Model\model167\kaggle --accelerator NvidiaTeslaT4`, then submit the notebook version.
- Local: `powershell -File Model\model167\run_reproduction.ps1` (RTX 4050: ~35 min for the 4 example movies,
  ~35 min validator, ~2.5 h post-process sweep). Expected validator base proxy 0.949047, selected tight55 0.951105.

## 2. Honest unseen-embryo testbed (model182, 186, 187)
1. Train a clean fold: `Other\.venv-gpu\Scripts\python.exe Model\model182\train_fold_windows.py --fold 0 --num-workers 0 --execute`
   with `$env:BIOHUB_RUNS_RESUMED=1` (about 3-4 h). Trained weights are kept in `Model\model182\weights\`,
   `Model\model186\weights\` (seeds 777, 31337), so retraining is optional.
2. Predict held-out movies with the exact 0.947 predictor: `Model\model180\predict_stems.py --stems ... --weights <ckpt>
   --env BIOHUB_SECONDARY_WEIGHTS=<ckpt2> [--env BIOHUB_TERTIARY_WEIGHTS=<ckpt3>]` (needs step 1 of section 1 to have
   materialized `C:\biohub_data\work\model167\tracking_repo`; third seed needs `Model\model187\patch_tertiary.py <repo>`).
3. Score any post-processing config: `Model\model180\sweep_configs.py --pred-root <dir> --stems ... --configs <json> --csv-out <csv>`.
   Wrapper scripts with the exact stems: `Model\model182\run_heldout.ps1`, `run_heldout_fold1.ps1`,
   `Model\model186\run_dualseed.ps1`, `Model\model187\run_tripleseed.ps1`.
Reference numbers (24 held-out 44b6 movies): one seed 0.79657, two seeds 0.80606, no-relink 0.78865,
synthetic-pretrained 0.75994.

## 3. Cloud-trained third seed (model185) and the 3-seed notebook (model188)
- Checkpoint: `Model\model185\weights\model185_scratch_all\edge_predictor_best.pth`
  (SHA-256 ac8dd16249dbffc08e8da88eb1cd7228ed10d1625c22d696774487ed5ef41357). Recreate: `Model\model185\make_strided_train.py`
  (13 GB stride-exact copy), rent with `Model\model185\vast.py`, run `Model\model185\remote_run_scratch.sh 130 250 8`.
- Notebook: `Other\.venv-gpu\Scripts\python.exe Model\model188\build.py <sha256>` writes `Model\model188\submission.ipynb`.
  It needs the checkpoint attached as a Kaggle dataset containing `model185_scratch_all/edge_predictor_best.pth` + `config.json`.

## 4. Synthetic data (model184, rejected)
`Model\model184\generate_local.py --n-seq 400` regenerates the sequences from the author's CC0 generator;
`convert_synthetic.py` converts them; `train_stage.py --stage pretrain|finetune` trains. Result: worse on the held-out embryo.

## 5. Communication helpers
`Other\scripts\agent_email.py send|inbox` (credentials in `Other\email.txt`).
