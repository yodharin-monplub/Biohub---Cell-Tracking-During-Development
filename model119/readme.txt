MODEL119 - DIVISION DAUGHTER-RECOVERY OPPORTUNITY AUDIT

Status: diagnostic complete; no scored model or frozen rescue rule.
Difficulty4/5 for opportunity audit; 0.97 CV target remains5/5.

Start from the new full model118 leader. Model118 preserves all model107 nodes
and all scored division true positives/FNs per movie; its only changes remove
weak extra daughter edges. Use the parity-verified model117 GT division-event
mapping to identify missed divisions with one correct direct daughter link and
all three cells present. Rebuild the model118 final graph, map event IDs to
the stable detector IDs, and read the saved original top-five neural candidate
scores. For each such true opportunity report candidate and incumbent parent
probabilities, physical daughter geometry, and temporal continuation.

The output is an oracle-labeled diagnostic, NOT an inference model and NOT a
full-score result. It does not assume unlabeled alternatives are negatives.
Do not choose a threshold from confirmation. If development lacks a strong
one-component signal, do not run a speculative expensive repair sweep.
Run: .venv-gpu/bin/python model119/audit.py (host permission for GEFF).
No cloud rental, Kaggle quota, upload, or sound.

RESULTS
The model117 event labels remain stable on model118: every movie has the
same division TP and FN after model118's false-fork veto. Among18 development
misses with all three cells present and one correct direct daughter link,
12 have a daughter already linked to another parent,16 have a top-five
neural alternative, but only one has alternative probability>=0.7 and none
reach0.8. Among14 confirmation misses,7 are occupied and all14 have a
top-five alternative, yet NONE has alternative probability>=0.7. For most
occupied positives the current wrong parent's neural score is far higher
than the true division link. The original model101 style strong-neural-evidence
reparent exception therefore cannot recover the remaining events at a useful
rate. Relaxing it to override 0.02-0.3 alternatives against 0.6-0.9 incumbents
would be speculative and high-risk without a stronger division-specific model.

Orphan-second-daughter opportunities remain:6 development and7 confirmation.
Their positive neural probabilities are often0.3-0.75, while parent distances
frequently exceed the existing7um gate. A future orphan proposal model might
use daughter appearance/temporal context, but this audit has not measured its
false-proposal population or demonstrated full-score gain. Do not claim it as
a candidate result. See audit.json for every event and its evidence.
