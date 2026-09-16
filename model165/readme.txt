MODEL165 - COMPLETED SECOND EMBRYO-HELD-OUT FOLD

Difficulty 5/5. Status: reciprocal fold trained, source calibrated,
exported, solved, repaired and organizer-scored. No Kaggle submission.

This is the missing reciprocal fold for evaluating the clean model156
backbone plus source-only association calibration. The fixed fold1
optimizer set is all71 44b6 movies; the outer evaluation set is all
128 6bba movies. It uses the same random-initialization training
configuration as fold0 (80 epochs x125 iterations, seed20260914,
math SDPA, batch2, no outer-label checkpoint selection). A SHA256-
ordered set of32 source44b6 movies is frozen for the calibrator:
24 fit, eight calibrator-dev. All selected movies are training-embryo
movies, disjoint from the128 outer6bba movies.

Run the CPU-only preparation with:
  .venv-gpu/bin/python model165/prepare_fold1.py

The preflight must not start the GPU trainer. The already-existing
model156/run_fold1.sh is the fixed fold1 trainer but MUST NOT be run
without explicit extension of the user's cumulative10-hour runtime
limit. The explicitly receipted model work through model164 totals
about9.53 hours before scoring overhead. Training fold1 alone took
about2.5 hours on the reciprocal local fold; exporting and evaluating
128 movies would require further hours. No cloud GPU is rented here.

After explicit approval of a cumulative runtime extension, the
guarded runner is model165/run_fold1_after_approval.sh. It requires
BIOHUB_RUNTIME_LIMIT_EXTENDED=1 and BIOHUB_RUNS_RESUMED=1 and must
only be used after approval of at least a16-hour cumulative limit.
It enforces a six-hour outer process timeout itself.
It trains fold1, exports32 source candidate graphs, and fits the same
source-only logistic calibration with the same AUC/top-parent gate.
Only if that gate passes does it export128 outer candidate graphs,
solve at p0.40, apply frozen repairs, officially score, and
aggregates fold0+fold1 movie-level counts without averaging the two
fold scores. It refuses overwrite or uninspected partial output.

No two-fold OOF/CV score can be reported until fold1 is trained,
source-calibrated, repaired and officially scored under the same
protocol, then aggregated with fold0. Even then, fold0 has been
reused for development and the frozen repairs were tuned on both
embryos, so the aggregate would be development CV, not fully nested
or untouched validation. Do not claim the0.97 target or submit from
this preparation alone.

Completed result (2026-09-15)
-----------------------------
The clean fold1 checkpoint trained for80x125 iterations on all71 44b6
movies, with all128 6bba movies excluded from optimization. Training
took9459.27s and produced checkpoint SHA256
90d0675494def7f2a3911f21ab1da98fe54924a4c95a558988d91bcfe376220a.
The source-only calibration gate passed (dev AUC0.969924->0.977567;
top1 1745/1980->1776/1980) before target inference. All128 target
movies were exported, solved and repaired with no score skips.

The official fold1 score is0.6685862464738727. Aggregating raw
organizer counts with model161 fold0 gives two-fold development CV
0.687418428513013 over199 movies (adjusted edge Jaccard0.684198393,
division Jaccard0.032200358, node recall0.924137074). The target0.97
was not reached. The large fold asymmetry (fold0 0.800115 versus
fold1 0.668586) shows severe embryo-domain shift and argues against
spending the next cloud run on the same single-embryo recipe.

The initial resumed postprocessing attempt failed only because
solve_target.py lacked the repository root on sys.path. All expensive
exports were preserved. The fixed host run also established that GEFF
Zstd decoding stalls inside the restricted sandbox, so solve/score was
run outside it. A120s per-movie ILP safety bound was recorded in the
solve contract; no movie hit it.
