MODEL196 - GENTLE FINE-TUNE OF THE PUBLIC PRIMARY (lr 1e-5, 12 epochs)

Hedge for model193 (lr 2e-5, 24 epochs): same recipe at half the learning rate and half the epochs (~4x smaller
total update), seed 20260925, all 199 train movies, warm start from the public primary (sha256 12f6881e...).
If model193 scores below 0.947 because the fine-tune drifted too far from the public weights/calibration, this
milder version is the next thing to try; if model193 helps, model197 (longer) is.

Run:    powershell -File Model\model196\run_train.ps1   (launch detached via WMI Win32_Process.Create)
Work:   C:\biohub_data\work\model196
Output: Model\model196\weights\model196_ft_primary\
Kaggle: reuse Model\model193\build.py pattern (a copy with 'model196' as the checkpoint path key).

2026-09-22 17:10: trained (sha256 47411ab79de00d6ff0fcc7a7c50f83b125c61c85a0e48c4f1ce306794be0ca9f); uploaded krittanutsomtuas/biohub-model196-ft-primary-weights; notebook built + monitor pinned. Push with Other\.venv\Scripts\python.exe Model\model196\kaggle\push.py only if model193's LB says a gentler fine-tune is worth trying.

2026-09-22 23:05: local check (model199 sweep2, 4 visible test movies @0.965): 123,543 nodes vs public 122,749; hard movie 44b6_0b24845f 20,590 vs public 20,707 (model193: 14,758). Keeps the public recall profile -> pushed kernel v1 (dummy-data commit).

2026-09-23 00:05: commit validated (242,731 rows vs 241k for the 0.947 baseline), submitted.
