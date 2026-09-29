MODEL207 - OWN DIVISION-EVENT DETECTOR TRAINED ON THE ANNOTATED DIVISIONS

Why: the public DivNet artifact (Kaggle dataset giorgosi/biohub-divnet-v2) claims out-of-fold AUC 0.845, but
re-implemented from its manifest it scores only 0.562 against the annotated divisions (Model\model206). Instead of
reverse-engineering it, this trains a detector of our own on the same idea.

Data (extract_crops.py -> C:\biohub_data\work\model207\crops.npz):
  one crop per annotated node: 4 image lags (t-1, t, t+1, t+2), shape (16, 32, 32) after pooling xy by 4, float16.
  Positives: the 151 annotated division parents in the 199 train movies. Negatives: annotated non-division nodes
  from the same movies and preferably the same frames (2,539 samples in total), so the net cannot win by
  recognising imaging conditions.

Model (train_divnet.py): small 3D CNN, 5 input channels (the 4 lags, percentile-normalised jointly, plus a
gaussian centre marker), class-weighted loss, 40 epochs, batch 32, lr 3e-4. Folds are grouped BY MOVIE (4 folds).

Result: out-of-fold AUC 0.907 (weights\summary.json; weights\fold0-3.pt, out-of-fold scores weights\oof.npy).

How it was used: Model\model208 wired the detector into the 0.947 pipeline (veto / rescue / rank / propose) and
Model\model209 submitted the "propose" variant (public LB 0.945 vs 0.947). On 40 train movies scored with the
official metric, proposing extra divisions gains +0.0018 (model208\official40_report.txt).

Reproduce:
  Other\.venv-gpu\Scripts\python.exe Model\model207\extract_crops.py
  Other\.venv-gpu\Scripts\python.exe Model\model207\train_divnet.py --data C:\biohub_data\work\model207\crops.npz
