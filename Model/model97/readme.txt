MODEL97 - FULL MODEL1 + THREE-FRAME LINK CONSISTENCY

Status: completed and REJECTED on2026-09-07. Baseline and candidate both
score0.9298357327505433; measured delta0.0 across all39 movies.
User hypothesis: use observations at t-1 and t+1 to improve interpretation at t.
Difficulty:4/5. Correcting identity mistakes without damaging valid motion or
divisions is the main challenge; existing model1 already smooths positions.

Exactly one new component, after full model1 repairs: exchange the middle
detections of two triplets only when both trajectories become more consistent.
For Aprev->A->Anext and Bprev->B->Bnext, propose Aprev->B->Anext and
Bprev->A->Bnext. No position smoothing, retraining or extra detections added.
All checkpoints, TTA, association, ILP, DeepCenter and prior repairs stay model1.
This starts from the baseline, NOT model95 or model96.

Frozen safeguards (config.json):
- Both middle nodes must have exactly one predecessor and one successor.
- Exclude entire graph components containing a division.
- Candidate middle nodes within6.5um; every new step at most7um.
- New midpoint error at most1.625um on each trajectory; improvement at least
  0.40625um on EACH trajectory, not merely a pooled improvement.
- Reciprocal best partner with0.40625um runner-up margin in total improvement.
- Proposals cannot overlap in any of their six context nodes.
- At most20 swaps per movie and0.05% of its node count, whichever is smaller.
- Preserve all nodes/coordinates, all degrees, divisions and total edge count.

Graph edits use the final rounded/clamped CSV coordinates. The embedded notebook
wrapper computes the same decision coordinates without changing original node
positions. control.ipynb is an exact full model1 copy; submission.ipynb changes
only cell14 to append this component. Original model1 remains untouched.

Run:
  .venv-gpu/bin/python model97/build_notebook.py
  bash model97/run.sh
Evaluate on the same39 development movies and organizer metric as
model92/local_rebuild/scored_baseline. CPU-only cached-graph test; no cloud GPU.
No GT in inference, family routing, threshold search, Kaggle upload or alarm.
Improvement on this reused development set requires separate confirmation;
neither an untouched reserve nor public-checkpoint training provenance is assumed.

Result:
- 579011 eligible middle nodes and19799 neighboring pairs inspected.
- Zero improving pairs met the frozen gates; zero accepted swaps.
- Output CSV is byte-identical to the input baseline CSV.
- All39 movie scores tie. This no-op does not qualify for promotion.
- Runtime77.36 seconds including official scoring; no alarm or cloud use.
- 14 unit/integration tests pass, including embedded notebook/CLI equivalence.
Receipts: results/repair_report.json, official_score.json, comparison.json,
status.json. Original model1 and all previous experiment outputs are unchanged.

Interpretation: this only tests reciprocal middle-detection swaps on final,
already-repaired/smoothed tracks. It does not test three-frame neural inference,
earlier candidate-link reranking, or all forms of temporal smoothing. Those
would require separately frozen experiments, not threshold tuning on test data.
