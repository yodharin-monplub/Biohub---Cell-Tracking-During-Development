MODEL 83 — FIVE-FOLD DELTA-SOUP TEST INFERENCE

Status
------
Complete and rejected by the exact visible-four gate. The output is strict
valid and backed up locally. It was not uploaded to Kaggle.

Purpose
-------
Run the unchanged primary production pipeline on all four competition test
movies using model82's five-fold checkpoint soup. The soup is the public
checkpoint plus one half of the mean fine-tuning delta from folds 0 through 4.

Frozen inference
----------------
- Checkpoint SHA256: 48d2c35ef724275e983832b4e7b10e86228472df9972cf5a67f76f3396deb950
- Detector threshold: 0.965
- Edge threshold: 0.5
- Native ILP costs: edge -1.0, appearance 0.0, disappearance 2.0,
  division 1.2
- Four-view XY test-time augmentation

Promotion gate
--------------
Require successful inference for exactly the four test datasets, strict CSV
validation, graph integrity validation, stable comparison against model1's
known 0.934 public-score submission, and a complete local backup. A better OOF
score raises confidence but does not guarantee a better Kaggle score.

Artifacts
---------
- run_cloud.sh: detached RunPod inference, conversion, and validation runner.
- geffs/: generated test tracking graphs (created on RunPod).
- submission.csv: candidate Kaggle CSV (created on RunPod).
- inference_receipt.json, conversion_receipt.json, validation_receipt.json:
  reproducibility and integrity evidence (created on RunPod).

Result
------
- Official visible-four score: 0.9312680913
- Frozen primary control: 0.9332567023
- Delta: -0.0019886110
- Submission rows: 235,050
- Submission SHA256:
  f372d51735e228cd17a228a092a97a46b890c49b7e6d0c0ff2e5c16481462fda

The soup improves two visible movies and regresses two. This indicates that
averaging the five fine-tuning deltas in weight space caused interference.
