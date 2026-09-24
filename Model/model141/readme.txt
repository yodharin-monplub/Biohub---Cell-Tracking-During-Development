MODEL141 - MOTHER-CELL PRE-DIVISION MORPHOLOGY PILOT

Status: train-only CPU feasibility experiment complete and rejected;
not a scored candidate or Kaggle submission. Difficulty4/5 for local experiment,
5/5 for0.97 paired CV.

Model139 showed nine of13 known orphan misses blocked by model1 neural
p<0.6 and four more blocked by the GT-space image classifier. Before
changing the proposal floor, test an independent division prior based
only on mother evidence at t-1 and t: physical movement and fixed
local image contrast, signal, spread, elongation, center offset and
their temporal changes. The signal is inference-visible; no future GT,
family ID, movie ID or cohort ID is a feature.

Use model136's121-train-only GT cases, selecting all81 continuous
positive divisions and up to five same-movie single-child negative
parents per positive, matched by time and daughter distance. Unknown
unannotated divisions are not used as negatives. Freeze the sample
before fitting, assign whole movies to five folds, and compare
regularized geometry-only against geometry+mother-image AP overall
and by family. A clear held-out gain is only a feasibility signal;
the actual detector-domain distribution and exact paired score would
still require a separate test. No cloud/Kaggle spend or sound.

Run: .venv-gpu/bin/python model141/pilot.py (host Zarr access)

RESULT
Across51 train-only movies, the fixed sample contains81 explicit
continuous division positives and387 same-movie single-child parent
negatives. Five-fold whole-movie regularized-logistic AP declined
from0.6101407 with time/motion/daughter-distance context to0.5842485
after adding mother morphology and t-1→t change features. Family44b6
rose0.5581128→0.6647459 but has only12 positives; larger6bba fell
0.6197477→0.5695564 with69 positives. The combined result does not
justify a mother-morphology proposal prior or a GPU rental. The full
detector-domain model140 capture remains the next decision source.
