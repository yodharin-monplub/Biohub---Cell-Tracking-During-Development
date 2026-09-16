MODEL118 - MODEL107 DIVISION FORK ATTRIBUTION AND SELECTIVE REPAIR

Difficulty 4/5 for this experiment; the 0.97 CV objective is 5/5.
Status: complete local paired experiment; eligible for review. A notebook is
packaged and one-video local GPU probability parity passed, but there is no
full hidden-test notebook run or verified public score.

Preserve original model1 and model107 byte-for-byte. The next component to
change is division handling only. Before selecting a rule, label every final
model107 fork with the exact organizer division scorer on development39 and
confirmation39. Require exact per-movie TP/FP/FN parity with saved full scores.
Measure whether each fork's two daughter links survived from the original
post-ILP graph, appeared in the neural top-five proposals, or were introduced
by downstream repair. Compare topology, physical distances, daughter
continuity, and neural support for the evaluable TP/FP forks. Unknown forks
are NOT negatives: sparse GT often cannot evaluate them.

The audit completed in 51.5 seconds with exact per-movie organizer parity on
all 78 movies. All 10 development TP forks have exactly one post-ILP daughter
link and repair-branch top-five probability >=0.429. Among 47 development FP
forks, 24 have repair-branch probability <0.4. All four confirmation TP forks
and 39/40 confirmation FP forks have the same one-post-ILP-link structure,
but confirmation outcomes were not used to choose the threshold. Unknown
forks are not negatives. In development the true-fork midpoint median is
1.88um versus 2.44um for FP, but one true fork is 5.69um; no geometry veto
is justified by this small sample.

Frozen model118 rule: after the unchanged full model107 output, inspect only
forks with exactly one daughter edge from the original ILP graph. If the
non-ILP daughter edge has captured top-five neural probability <0.4, remove
that edge. Keep every node and every other edge. If the probability is
missing, make no edit. This is a division-specific conservative veto, not a
global probability replacement or a relinking rule. The threshold and
promotion gate are written in config.json before scoring the candidate.

Run read-only audit with host permission because the restricted sandbox stalls
while reading GEFF: .venv-gpu/bin/python model118/audit.py
Run full candidate: bash model118/run.sh (with host permission for scoring).
Outputs: model118/audit.json, model118/results/{development,confirmation}/,
and model118/results/comparison.json.

Decision gate: require positive complete organizer score gain versus model107
on BOTH 39-movie cohorts; no family loss worse than 0.001; no node-recall loss
above 0.001; exact validation and movie coverage. Both cohorts have already
been reused for earlier experiments, so success is not untouched CV and not
a public-leaderboard claim. No public submission, cloud spend, Kaggle GPU, or
audible alert in this experiment.

RESULTS (exact organizer scorer, all 39 movies in each reused cohort)
The frozen veto removed 707 development and 842 confirmation daughter edges;
nodes and every other edge remained unchanged. Output CSVs validated and
their hashes match the exact scorer inputs. Development: model107 0.9530565302
to model118 0.9577408655 (+0.0046843354). Confirmation: model107
0.9495462814 to model118 0.9525886848 (+0.0030424034). Both families gain
in both cohorts: development 44b6 +0.007094, 6bba +0.004108;
confirmation 44b6 +0.007979, 6bba +0.001743. Node recall is unchanged.
Adjusted edge Jaccard rises from 0.9415623 to 0.9421159 development and
0.9434857 to 0.9440780 confirmation. Division Jaccard rises from 0.1149425
to 0.15625 development (10 TP / 24 FP / 30 FN) and 0.0606061 to 0.0851064
confirmation (4 TP / 21 FP / 22 FN). All predeclared paired promotion gates
pass. This is the new best local score, but still below the 0.97 CV target.

DEPLOYMENT PREFLIGHT
build_notebook.py packages submission.ipynb from byte-frozen model107, changing
only cells12 and14. Cell12 captures the existing fused top-five neural scores
to a per-movie sidecar without changing predictor/ILP decisions. Cell14 applies
the exact frozen0.4 veto after unchanged full model107 repair, before CSV
writing. test_packaging.py passed: the capture hook equals the original
model100 operator and the packaged veto removes exactly the47 scored offline
edges on one development movie. An initial GPU preflight failed because its
test harness omitted model1's bidirectional-fusion environment (detector
coordinates still matched); that failed receipt is preserved. The corrected
preflight restored every saved model1 BIOHUB setting and on local RTX4050
matched 5,048 detector coordinates, 24,975 captured candidate endpoints, and
all neural probabilities exactly (maximum difference0) on6bba_76db78c1.
See preflight_gpu_corrected.json. No cloud or Kaggle GPU was used. This is
strong packaging evidence, not a full notebook run or public-score proof.
