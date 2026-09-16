MODEL 6 — DENSE-SYNTHETIC DETECTOR-HEAD FINE-TUNE

Status
------
Kaggle training versions 1 and 2 completed on explicitly requested T4 x2
hardware. Version 2 evaluated every emitted blend over a fixed threshold
sweep. No competition score and no claim of real-domain improvement yet.

Version 1 evidence
------------------
- Frozen-head held-out synthetic top-count recall at 3.5 um: 0.82339.
- Trained-head held-out synthetic top-count recall at 3.5 um: 0.83499.
- Fixed-threshold 0.965 recall fell from 0.86085 to 0.80835 while precision
  rose from 0.98961 to 0.99278. This means alpha 1.0 is not automatically the
  best inference checkpoint.
- The four full state dicts have checksum receipts in output-v1.
- Direct comparison of all 136 serialized tensor storages found exactly two
  changed storages, corresponding to detect_head.weight and detect_head.bias.

This is useful synthetic evidence, not a promotion result. Version 2 is exactly
reproducible: all four checkpoint SHA-256 values match version 1 byte for byte.

Selected synthetic operating point
----------------------------------
The first real-domain candidate is alpha 0.50 with threshold 0.950. Against
the frozen production operating point (base head, threshold 0.965), it keeps
the mean count ratio essentially fixed while improving every measured center
diagnostic:

metric                              base 0.965      alpha-50 0.950
mean predicted/truth count ratio    0.889582        0.889427
precision within 7 um               0.989607        0.990959
recall within 7 um                  0.860849        0.861704
top-count recall within 3.5 um      0.823387        0.829197
top-count recall within 7 um        0.846100        0.850014

Alpha 1.0 remains rejected as too disruptive at the production threshold.
The alpha-50 choice is only a candidate until it passes real data validation.

Hypothesis
----------
The real annotations cover only a small fraction of nuclei, so the original
detection BCE eventually treats many genuine but unlabeled nuclei as negative.
The public CC0 sequences are fully labeled. Fine-tuning only the secondary
model's 1x1x1 detection head can learn from dense targets without disturbing
its real-image feature extractor or any association-transformer weight.

Controlled design
-----------------
- Start from checksum-pinned secondary seed weights.
- Use 100 fully labeled sequences with a fixed 80/20 sequence split.
- Freeze TemporalUNet3D and every transformer tensor.
- Train only detect_head.weight and detect_head.bias (33 parameters).
- Select an epoch using count-matched center recall within 3.5 micrometers.
- Emit 25%, 50%, 75%, and 100% head blends toward the synthetic optimum.

The conservative alpha-25 output is the first real-data candidate. All other
state-dict tensors are copied unchanged. The training kernel is separate from
the code-competition submission kernel.

Promotion gate
--------------
1. Training must beat the frozen head on sequence-held-out dense synthetic
   center recall and produce checksum/invariant receipts.
2. Choose blend strength and detection threshold only on embryo-held-out real
   movies using the official metric.
3. Reject a blend if estimated-node ratio exceeds 1.05 on any validation movie,
   either embryo family loses over 0.002, or hidden-test runtime is affected.
4. Only then attach the selected checkpoint to a copied model1 submission.

Kaggle execution
----------------
.venv/bin/kaggle kernels push -p model6 --accelerator NvidiaTeslaT4
