MODEL 82 — FIVE-FOLD PRODUCTION DELTA SOUP
==========================================

Status
------
Complete. All five fold-specific checkpoints and the five-fold delta soup were
created on the RTX A6000 and copied locally with verified hashes. No Kaggle
upload was performed.

Evidence before scaling
-----------------------
- Fold 0: 0.9032503 -> 0.9041413, +0.0008910, 22 wins / 17 losses,
  node-recall delta -0.000579.
- Untouched fold 1: 0.9469801 -> 0.9527912, +0.0058112,
  27 wins / 12 losses, node-recall delta -0.000403.

Protocol
--------
- Train the frozen three-epoch, 250-iteration full-model recipe independently
  on folds 2, 3, and 4. Fold-0 and fold-1 checkpoints already exist.
- Preserve every fold-specific tuned and 50/50 interpolated checkpoint.
- Build one production checkpoint as:

    public + 0.5 * mean(fold_i_tuned - public), i=0..4

  This equals the average of the five fold-specific 50/50 blends while keeping
  one-model inference cost for Kaggle.
- Preserve public values for discrete buffers and save hashes/provenance for
  all six source checkpoints and the production soup.

No Kaggle upload or submission is performed by this directory.

Result
------
The soup checkpoint is valid with SHA256
48d2c35ef724275e983832b4e7b10e86228472df9972cf5a67f76f3396deb950.
Model83's exact visible-four test scored 0.9312681 versus the frozen primary
control at 0.9332567, so the soup was rejected as a production replacement.
Model84 subsequently evaluates each fold-specific half-step blend separately.
