#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
split="${BIOHUB_SPLIT:-0}"
eval_name="${BIOHUB_EVAL_NAME:?Set BIOHUB_EVAL_NAME to a unique immutable run name}"
checkpoint="${BIOHUB_CHECKPOINT:?Set BIOHUB_CHECKPOINT to an edge_predictor_best.pth path}"
eval_base="${BIOHUB_EVAL_BASE:-${workspace}/model77/evaluations}"
eval_root="${eval_base}/${eval_name}/split_${split}"

cd "${workspace}"
checkpoint_sha="$("${python_bin}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${checkpoint}")"

"${python_bin}" scripts/export_primary_topk_candidates.py \
  --repo "${workspace}/data/public/support-pack/repo" \
  --weights "${checkpoint}" \
  --expected-weight-sha256 "${checkpoint_sha}" \
  --data-dir "${data_dir}" \
  --splits "${workspace}/model77/cloud_splits.json" \
  --split "${split}" \
  --det-threshold 0.965 --edge-threshold 0.000001 \
  --parents-per-target 5 --max-edge-distance 12 \
  --output-dir "${eval_root}/candidates" \
  --receipt "${eval_root}/candidate_receipt.json" \
  --resume

"${python_bin}" scripts/sweep_topk_parent_ilp.py \
  --candidate-dir "${eval_root}/candidates" \
  --output-dir "${eval_root}/solved" \
  --edge-thresholds 0.40 \
  --edge-weight -1.0 --appearance-weight 0.0 \
  --disappearance-weight 2.0 --division-weight 1.2 \
  --num-threads 8 --receipt "${eval_root}/solve_receipt.json" --resume

if [[ ! -f "${eval_root}/gap_receipt.json" ]]; then
  "${python_bin}" scripts/close_track_gaps.py \
    --input-dir "${eval_root}/solved/edge_p_0p400" \
    --output-dir "${eval_root}/gap_geffs" \
    --radius-grid 5 --min-acceleration-um 6.5 --require-internal \
    --receipt "${eval_root}/gap_receipt.json"
fi

"${python_bin}" scripts/geffs_to_submission.py \
  --geff-dir "${eval_root}/gap_geffs" \
  --output "${eval_root}/oof.csv" \
  --report-json "${eval_root}/conversion_report.json"

"${python_bin}" scripts/score_submission.py "${eval_root}/oof.csv" \
  --train-dir "${data_dir}" \
  --json-out "${eval_root}/official_score.json"

echo "Exact fold score: ${eval_root}/official_score.json"
