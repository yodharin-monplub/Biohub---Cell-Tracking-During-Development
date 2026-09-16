#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
checkpoint="${workspace}/model77/cloud_outputs/pilot_fold0_seed_20260906/split_0/edge_predictor_best.pth"
root="${workspace}/model78/evaluations/detector_sweep"
thresholds=(0.9638 0.9625 0.9613)

cd "${workspace}"
ulimit -n 65535 2>/dev/null || true
checkpoint_sha="$("${python_bin}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${checkpoint}")"

"${python_bin}" scripts/sweep_primary_thresholds.py \
  --repo "${workspace}/data/public/support-pack/repo" \
  --weights "${checkpoint}" \
  --expected-weight-sha256 "${checkpoint_sha}" \
  --data-dir "${data_dir}" \
  --splits "${workspace}/model77/cloud_splits.json" \
  --split 0 \
  --thresholds "${thresholds[@]}" \
  --edge-threshold 0.000001 --parents-per-target 5 --max-edge-distance 12 \
  --skip-ilp --resume \
  --output-dir "${root}/candidates" \
  --receipt "${root}/candidate_receipt.json"

for threshold in "${thresholds[@]}"; do
  det_slug="det_${threshold//./p}"
  eval_root="${root}/${det_slug}"
  "${python_bin}" scripts/sweep_topk_parent_ilp.py \
    --candidate-dir "${root}/candidates/${det_slug}" \
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

  "${python_bin}" scripts/compare_cloud_pilot.py \
    --baseline "${workspace}/model77/evaluations/public_baseline/split_0/official_score.json" \
    --pilot "${eval_root}/official_score.json" \
    --output "${eval_root}/comparison.json"
done

echo "Model78 detector sweep complete: ${root}"
