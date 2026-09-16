MODEL135 - RAW IMAGE CONTRAST AUDIT FOR DIVISION PROPOSALS

Status: diagnostic complete; no new scored model. Difficulty4/5 audit,
5/5 target0.97 CV.

Model134 shows model133 added1 scored TP/15FP in development and5TP/
10FP in confirmation, while neural/trajectory feature distributions
overlap. Before training a division-specific image model on only85
train-only positives, extract simple local fluorescence features from
the actual Zarr frames around exactly those scored TP/FP mother/daughter
triples. Use a fixed5x13x13 voxel patch, local shell background, a
3x5x5 core, and normalized contrast/signal ratios. Compare summaries;
do not treat unscored/unknown forks as negatives or tune a threshold
from six positives. This is a feasibility check for image information,
not an inference rule, score, public test, or cloud request.

Run: .venv-gpu/bin/python model135/audit.py (host Zarr access)

RESULT
The audit extracted all31 organizer-scored new model133 forks: development
1TP/15FP and confirmation5TP/10FP. Simple fixed-patch fluorescence
summaries do not transfer cleanly. For example, orphan peak contrast is
lower in the lone development TP than the development FP median
(114.5 versus289.5), but higher in confirmation TPs than its FP median
(477 versus251.5). Among6bba examples, its positive-vs-negative AUC is
0.25 in development and0.925 in confirmation. Daughter-sum/parent
signal likewise reverses (development TP1.59 versus FP median1.76;
confirmation TP median3.44 versus FP median1.84). This sample is too
small and selected to tune a brightness threshold. See audit.json.
The next experiment should test temporal shape/persistence with train-only
positives and hard negatives, holding out movies; no new rule is promoted.
