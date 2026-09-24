MODEL99 - FULL MODEL1 DIVISION LOSS AUDIT

Difficulty:4/5. Sparse labels and stage-specific matching make attribution
difficult. This is a diagnostic, NOT a new tracking submission.
Status: completed2026-09-07. Original model1 and prior experiments unchanged.

Compare the fixed39 movie fullmodel1 baseline at two saved stages:
1. Saved GEFF after neural link selection and ILP, before notebook repairs.
2. Final repaired integer CSV, already organizer-scored at0.9298357327505433.
Recompute organizer division TP/FP/FN and require final per-movie counts to
match the baseline receipt. Model1 public0.934 remains a separate evaluation.

For each GT division, use organizer-local distance matching (7um, anisotropic
voxel scale). Report parent/daughter presence and direct fork topology at
both stages, annotated context coverage, and current daughter parent links.
Official success allows temporal offsets/grandchild evidence: direct mapped
fork topology is a diagnostic, NOT a replacement for the organizer metric.

For final misses with exactly one correct parent-daughter edge, report
hypothetical safe-division gates on FINAL coordinates/graph. This does not
claim the same gates failed at the historical pre-repair stage. Caps and
competing proposals are not reconstructed. No ground truth enters inference.

IMPORTANT LIMIT: cached GEFFs are POST-ILP. ILP can remove nodes too. Missing
nodes cannot be attributed to the detector alone, and missing edges cannot
be separated into scorer/threshold/greedy/ILP causes without a new instrumented
inference run. Do not mislabel these files as raw neural candidate graphs.

Run: .venv-gpu/bin/python model99/monitor_run.py
Quiet ten-minute supervision; no alarms. CPU cached-graph diagnostic, no cloud,
training, uploads, submission edits, threshold search or GT-driven graph fixes.
39 movies are reused development data, not an untouched holdout.

RESULTS
All39 final per-movie division TP/FP/FN counts exactly match the saved organizer
receipt.40 annotated division events audited. Runtime62.57 seconds including
supervisor; four unit tests pass. No active model99 run remains.

Stage                          Division TP / FP / FN
Post-ILP, before repairs        0 /0 /40
Final complete model1           7 /41 /33
All7 successful recoveries appear during notebook repairs. No previously
successful post-ILP event was lost (there were none).

Final33 missed events:
-21: parent and both daughters locally matched; exactly one correct direct link.
-11: at least one parent/daughter lacks a distinct local match. This includes
  ambiguous/shared nearby detections; it does NOT prove11 detector failures.
-1: all three locally matched, neither correct direct parent-daughter link.

Among the21 one-link misses, overlapping hypothetical final-state gate failures:
-14: second daughter already has another incoming link (not an orphan).
-18: second daughter is beyond7um from parent.
-17: separation growth below2.25um.
-14: second daughter is not nearest orphan.
-6: sister distance above12um.
These are not historical rejection counts and must not be summed as distinct
events. They show why simply rescoring the existing orphan-only proposal set
cannot address many of the observed missed divisions.

ILP MECHANISM CHECK
Original source and rebuild_control.log confirm edge cost=-probability,
appearance cost0, disappearance cost2, division cost1.2. Removing one of a
fork's two outgoing edges and starting that daughter independently changes
cost by p-1.2 <= -0.2, retaining all nodes and downstream tracks. Consequently
the optimum cannot contain a true two-child fork under these baseline costs.
verify_ilp.py confirmed this on a13-node synthetic lineage: baseline selected
11 edges/no fork; synthetic division-cost0.5 control selected12 edges/one fork.
0.5 is a mechanism test ONLY, not a selected competition hyperparameter.

This cost issue was already noted in model17's PRIMARY-ONLY experiment; the
new evidence here is its observed impact in the complete original-model1
39-movie pipeline. Earlier model17 cost reductions could not recover missing
candidate topology. Model52 top-k/global-cost reduction produced many forks.
Therefore do NOT repeat a blind global division-penalty reduction.

NEXT TARGET
Instrument an unchanged fullmodel1 run to retain pre-ILP candidate evidence,
including low-ranked alternative parents. First verify unchanged control
predictions. Then assess selective division/reparenting proposals that can
compete with a wrong existing incoming link, using association confidence and
temporal evidence. Such a change must be separately frozen, sparse-label-safe,
and evaluated with complete model1 plus exactly one changed component.
No score improvement or prize claim follows from this diagnostic.

Artifacts: audit.json, events.json, ilp_mechanism.json, status.json, run.log.
Original public model1 remains0.934; local39 baseline remains0.9298357327505433.
