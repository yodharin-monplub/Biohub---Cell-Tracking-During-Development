MODEL203 - HALF-GENTLE FINE-TUNE OF THE PUBLIC PRIMARY (lr 1e-5, 6 epochs)

Fills the gap between model196 (lr 1e-5, 12 ep; keeps the public recall, submitted) and model201 (lr 5e-6, 8 ep).
Same warm start from the public primary, all 199 movies, seed 20260929. Checked locally (model199 sweep) before any
Kaggle push; the aim is the largest update that still keeps the hard-movie recall (44b6_0b24845f, public 20,707).

Run:    powershell -File Model\model203\run_train.ps1  (detached via WMI; waits for the GPU)
Output: Model\model203\weights\model203_ft_primary\

2026-09-23 02:40: local check @0.965: total 125,869 nodes, hard movie 23,052 (+11% vs public 20,707). Same 'more detections' zone as model201 (+15%); model196 (-0.6%) is the crossing point. Held - no more variants until model196's LB score arrives.
