MODEL169 - ONE-DIMENSIONAL TIGHT MOTION-RADIUS REFINEMENT

Difficulty 4/5. Status: complete locally; not submitted.

Control: the exact portable model167 public0.947 pipeline, including all
checkpoints, detection/association TTA, ILP, DeepCenter gates, graph repairs,
validator movies and metric. Change one component only: refine the already
selected MOTION_RELINK_TIGHT_UM value around5.5 using a predeclared grid
5.0,5.25,5.5,5.75 against the unchanged6.0 base. All unrelated public
candidate knobs are omitted from this refinement because model167 already
rejected them.

The run must reuse byte-identical cached test and validator prediction graphs
from model167, recompute only postprocessing candidates, and require the same
selection margin. It may support a later independent confirmation but uses the
same eight tuning movies, so its selected proxy is not CV and cannot prove
Public LB. Never alter model1 or model167 outputs.

RESULT (2026-09-15)
The exact cached run completed with all four candidates and selected tight525,
MOTION_RELINK_TIGHT_UM=5.25. The unchanged base proxy was0.9490489368;
tight525 reached0.9523628978, a gain of0.0033139611 over base and
0.0012562701 over model167's tight55. Candidate5.0 scored0.9464903662,
5.5 scored0.9511066278, and5.75 scored0.9496694746. The final CSV contains
241,293 rows and has SHA256
538c16d801b0a93e3716705f30375a57441b1480fc82cae6fed7b85c16888792.
Independent validation passed all four test movies,400 frames, contiguous IDs,
node/edge schema, t-to-t+1 edge timing, max indegree1 and max outdegree2.

This score is still a selector proxy on the same eight movies, not honest CV.
Model170 audits its cross-family and leave-one-movie-out robustness. Do not
submit model169 before comparing that evidence and the pending model167 LB.
