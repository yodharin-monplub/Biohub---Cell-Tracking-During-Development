MODEL195 - FINE-TUNED PAIR: model193 (fine-tuned public primary) + model194 (fine-tuned public secondary)

Only to be pushed if model193 beats 0.947 on the public LB (fine-tuning the primary alone helps). Same notebook
blocks as model193 (commit = dummy-data check with the validator sweep disabled; full pipeline in the scoring rerun;
read-only symlink fix), plus BIOHUB_SECONDARY_WEIGHTS -> the attached model194 checkpoint (hash-checked).

Build:   Other\.venv\Scripts\python.exe Model\model195\build.py <model193_sha256> <model194_sha256>
         (model193 sha256 = 710a6379475fc26dd44509846c37ff7ea58901fece496be12809e160bb7cc127)
Datasets: 3 pilkwang public + krittanutsomtuas/biohub-model193-ft-primary-weights + a model194 weights dataset
Status 2026-09-22: build.py written and test-built; waiting for model194 training and model193's LB score.

2026-09-22 10:35: model194 trained (sha256 509c2222...5b5e) and uploaded as krittanutsomtuas/biohub-model194-ft-secondary-weights; kaggle\ package built (kernel krittanutsomtuas/biohub-model195-ft-pair). Push with: Other\.venv\Scripts\python.exe Model\model195\kaggle\push.py  -- only if model193 beats 0.947.

2026-09-22 11:00: pushed v1 without waiting for model193's LB (scoring takes ~16 h; running both in parallel saves a day; the dummy-data commit costs ~25 GPU-min).

2026-09-22 11:40: commit v1 COMPLETE + validated (227759 rows), submitted.
