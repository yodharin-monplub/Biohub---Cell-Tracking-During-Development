MODEL115 - MODEL107 WITH HALF-SIZED ILP DISAPPEARANCE PENALTY

Status: launch failed before scoring; no model115 score. Difficulty5/5 for 0.97 CV target. This was
one upstream ILP component change from the exact complete model107 system:
BIOHUB_ILP_DISAPPEARANCE_WEIGHT 2.0 ->1.0 in cell4. The value is frozen
before any model115 organizer score. All detector weights, two-seed fusion,
8-view TTA, candidate probability floor0.48, edge/appearance/division ILP
costs, model107 additive tight free-endpoint motion, DeepCenter, gap/division
repairs, short-track filtering/rescue and output format stay unchanged.

Mechanism: with appearance cost0 and edge reward -p, reducing the fixed
track-end cost may let ILP retain shorter but high-probability true tracks.
model113 found 140/304 development and156/309 confirmation unique missing
GT endpoints have a raw detector match absent from post-ILP. model112 showed
that simply disabling final short-track filtering hurt both full scores;
model115 retains that filter and changes only the upstream solve. This could
still add false tracks and regress the adjusted node-count metric.

Offline evaluation is allowed ONLY if model114 reconstructs all39 original
post-ILP graphs exactly on EACH cohort from saved top-five captures. At run
time, verify original model1/model107 hashes and all model114 parity receipts.
Preflight one development and one confirmation movie: reconstruct original
cost2 ILP, execute unchanged complete model107 repair, and require exact
final nodes/rounded positions/edges versus model107's scored CSV. A mismatch
stops BEFORE writing a candidate. Then solve the same saved candidate graph
with cost1, replay complete repairs, validate strict CSV and score through
the exact organizer metric on development39 AND confirmation39 regardless of
first-cohort outcome. Compare with model107 and original model1 on identical
movies. Promote for review only if both cohorts improve over model107, each
family loses <=0.001, and mean node recall loses <=0.001. No threshold sweep,
family routing, GT at inference, Kaggle or cloud action.

Both cohorts have been reused and public-checkpoint training provenance is
not proven; these are local comparisons, not untouched CV or public score.
The first preflight checks graph/repair parity, but cannot make these reused
labels independent. Original model1/model107 data are never edited.

Run after both model114 parity receipts pass:
  .venv-gpu/bin/python model115/build_notebook.py
  .venv-gpu/bin/python model115/monitor_run.py

Local RTX4050 only for existing DeepCenter repair; no detector rerun or
Vast.ai spend. Quiet supervisor checks every600 seconds, no alarm.

LAUNCH FAILURE
The first run stopped before the cost2 control preflight because model1 cell6
enforces the original disappearance value2.0. Changing only cell4 to1.0
triggered the intended configuration-drift guard. No candidate CSV, score,
or Kaggle/cloud action was produced. Preserve this failed evidence. The
corrected model116 changes the same single conceptual ILP cost in both its
setting and its integrity guard; model115 is not a rejected score result.
