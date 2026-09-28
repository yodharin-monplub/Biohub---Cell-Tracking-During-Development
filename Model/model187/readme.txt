MODEL187 - THIRD SEED IN THE 0.947 PIPELINE

Difficulty 4/5. Status: patch written; honest held-out test queued; no submission yet.

Why: model186 showed that a second independent seed is worth +0.0095 on an unseen embryo, the
only positive lever measured on held-out data. The public pipeline already fuses two seeds; the
model185 cloud model (all 199 movies, random init) can be a third.

How (patch_tertiary.py): an idempotent text patch of the pipeline's patched predictor. With
BIOHUB_TERTIARY_WEIGHTS set, the "secondary model" becomes an ensemble of the public secondary
and the tertiary checkpoint: UNet features concatenated on the channel axis, detection logits
averaged after per-frame standardisation, edge logits averaged over the members' own
transformers. The primary model, mean/std alignment, retention guard, D4 feature TTA,
low_margin_consensus fusion, ILP and post-processing are untouched. Env unset = identical
behaviour to the 0.947 notebook.

Honest test (run_tripleseed.ps1): three clean fold0 seeds (20260914, 777, 31337), each trained
on 128 x 6bba movies only; primary = seed 1, secondary = ensemble(seed 2, seed 3); same 24
held-out 44b6 movies. Reference points: one seed 0.79657, two seeds 0.80606.
Promotion: if three seeds >= two seeds on held-out, build the Kaggle notebook with the cloud
checkpoint as tertiary (needs the user's OK to upload the weights as a private Kaggle dataset).
Cost: about +50% inference time per movie (one more UNet + 8-view TTA).


RESULT (2026-09-20)
-------------------
Three seeds 0.80202 vs two seeds 0.80606 vs one seed 0.79657 on the 24 held-out 44b6 movies (details in score.txt,
per-movie numbers in results\heldout_fold0_tripleseed.per_movie.csv). Paired comparison is a 12/12 split and the
edge TP/FP/FN totals differ by <2%, so the third seed is neutral: the gain from seed averaging is already
saturated at two seeds with this fusion (secondary-side feature concat + averaged logits). Six extra false
divisions (17 vs 11) account for about half of the proxy difference.
