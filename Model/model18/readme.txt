MODEL 18 — DENSITY-ADAPTIVE PRIMARY DETECTOR THRESHOLD

Status
------
Complete local composition audit and current visible-score champion. Strict
validation passes. Official visible-four score 0.9337902030, a gain of
+0.0005335008 over fixed-threshold model15. This is not yet a Kaggle notebook.

Hypothesis
----------
Model16 found a consistent density interaction: threshold 0.980 improves both
44b6 movies and the small 6bba movie, while it harms the 68k-node dense 6bba
movie. Use the competition-provided estimated node-count metadata to choose one
of two frozen whole-movie pipelines:

- estimated nodes below 50,000: detector threshold 0.980;
- estimated nodes at least 50,000: detector threshold 0.965.

This is a label-free rule. Dataset names, prefixes, ground-truth edges, and
visible metric values are not runtime inputs. The threshold is selected once
per movie, preserving a coherent detection/association/ILP graph.

Frozen controls
---------------
- Primary weight SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Four-view XY TTA, edge softmax threshold 0.5.
- ILP costs: edge -1.0, appearance 0.0, disappearance 2.0, division 1.2.
- No ensemble or graph post-processing.

Promotion gate
--------------
1. Strict graph validation and the official four-movie metric must pass.
2. The aggregate must beat fixed 0.965 model15.
3. Audit the same fixed density boundary across a broader embryo-held-out set
   before treating the visible gain as generalization evidence.
4. The production notebook must read only allowed test metadata and must not
   contain visible dataset IDs.

Gate interpretation
-------------------
The aggregate gain is entirely node-count adjustment: labeled edge matches are
unchanged and aggregate raw edge Jaccard remains 0.9303912648. Treat this as a
provisional calibration gain; validate the fixed 50,000-node density boundary
across more embryos before production promotion.
