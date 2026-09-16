MODEL 7 — CONFIDENCE-AWARE REAL-DOMAIN TEACHER

Status
------
Private Kaggle version 1 is running on explicitly requested T4 x2 hardware.
This model does not write or submit a competition submission.

Hypothesis
----------
The secondary TemporalUNet detector is fast enough for the hidden rerun but
was trained from sparse annotations. FOCUS-3D is a strong general 3D nuclei
segmenter with dense instance predictions, but running it beside the baseline
on every hidden frame would exceed the 12-hour budget and its raw counts are
not stable on every embryo. Use it only offline as a real-image teacher, then
distill carefully filtered centers into the small baseline detector head.

Experiment
----------
- Run FOCUS-3D on four real 100-frame movies spanning both embryo prefixes.
- Preserve every raw instance center plus mean/max confidence, voxel count,
  mean intensity, nearest temporal support, mutual-nearest support, and its
  distance to a model1 node in the same frame.
- Compare candidate gates at confidence 0.60/0.70/0.80/0.90, one- and
  two-sided temporal support, and novelty relative to model1.
- Report sparse-truth node recall and predicted/estimated node-count ratios
  per movie. Sparse annotations cannot estimate precision, so no filter is
  promoted from recall alone.
- Record source/checkpoint checksums and abort unless both T4 devices exist.

Provenance
----------
FOCUS-3D source: https://github.com/yu-lab-vt/FOCUS-3D (BSD-3-Clause).
Official weights: https://huggingface.co/Qinghua-thu/FOCUS-3D (Apache-2.0).
The Kaggle mirror used by this job is qiweiyin/focus3d-nuclei-runtime.

Promotion gate
--------------
1. Zero failed frames and reproducible checksums.
2. A filter must preserve model1-level sparse-truth recall in both embryo
   prefixes while avoiding implausible estimated-node ratios.
3. Novel pseudo-labels must have temporal support; isolated teacher-only
   detections are never training targets.
4. Model8 must validate a conservative head blend before model7-derived data
   can enter a competition notebook.

Kaggle execution
----------------
.venv/bin/kaggle kernels push -p model7 --accelerator NvidiaTeslaT4
