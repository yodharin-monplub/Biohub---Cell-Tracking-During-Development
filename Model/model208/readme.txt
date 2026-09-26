MODEL208 - DIVISION GATE: WIRING THE model207 DETECTOR INTO THE 0.947 PIPELINE

Local ground-truth evaluation on the notebook's own 8 validator movies (identical predictions in every row; only
the division logic differs; each train movie scored with the fold model that never saw it):

  mode          proxy       div_jaccard  div_tp/fp/fn   note
  off           0.9490468   0.2308       3 / 1 / 9      unchanged 0.947 pipeline
  veto  <0.15   0.9490468   0.2308       3 / 1 / 9      identical - there is almost nothing false to remove
  rank          0.9490464   0.2308       3 / 1 / 9      neutral: the per-frame cap never binds (cap_skipped = 0),
                                                        so every surviving candidate is already added and
                                                        re-ordering them cannot change the output
  rank_bypass   0.9472426   0.2143       3 / 2 / 9      WORSE: overriding the DeepCenter veto added 18 divisions
                                                        (140 -> 158), none of them correct

Conclusion: the pipeline recovers 3 of 12 annotated divisions and invents only 1, so the entire opportunity is the
9 it MISSES - and those are not in the candidate pool. They are cut before ranking by the geometric filters
(mutual-NN and divergence reject 62-273 and 545-2648 candidates per movie) or never proposed at all. A better
ranker cannot fix a candidate set that lacks the right pairs, and loosening the DeepCenter veto only adds noise.

The detector itself is sound (model207: movie-grouped CV AUC 0.907, and 0.880 through this code path). The binding
constraint is candidate generation, not scoring.

Files: build.py (patches: veto / rescue / rank / deepcenter bypass), div_gate.py (fold-correct scoring),
compare_in_process.py + run_compare2.ps1 (predict once, score many), check_gate.py, gate_comparison.json.

2026-09-25 40-MOVIE CONFIRMATION (20 per embryo type, 60 annotated divisions; gate_comparison_val40.json):
  off      0.916723  13 / 28 / 47
  p40n30   0.918214  18 / 50 / 42   (= model209)
  p30n50   0.918007  18 / 53 / 42
Gain holds (+0.0015) but is much smaller than on the 8 movies (+0.0063); added-division precision ~19% vs ~17% break-even.
Stricter settings running as two parallel processes: run_val40_a.ps1 (0.50/30, 0.60/30), run_val40_b.ps1 (0.40/10, 0.50/10).

2026-09-26 OFFICIAL-METRIC 40-MOVIE COMPARISON (run_off40_all.ps1 -> compare_in_process.py, one prediction, six
variants; CSVs clamped at 0 and scored by rescore_official.py; full table official40_report.txt, raw
gate_comparison_official40*.json):
  variant   official  delta     adjEdgeJ  div tp/fp/fn
  off       0.91506   +0.00000  0.90060   12/23/48    unchanged 0.947 pipeline (model167)
  nosafe    0.90071   -0.01435  0.90071    0/0/60     extra 'safe' divisions are worth ~0.014 - keep them
  veto40    0.91506   +0.00000  0.90060   12/23/48    veto removes nothing
  veto15    0.91506   +0.00000  0.90060   12/23/48    veto removes nothing
  p40n30    0.91685   +0.00179  0.90050   17/44/43    = model209 (public LB 0.945 vs 0.947)
  p60n30    0.91732   +0.00226  0.90065   17/42/43
Both propose variants gain the same 5 correct divisions (one each on 5 different movies) for ~20 false ones; the edge
term is unchanged. p60n30 vs p40n30 is 2 false divisions on 40 movies = noise, so no model210 was built.
DECISION: final slot 1 = model167 (0.947, submission 56256610); final slot 2 = model209 (p40n30, submission
56518575): best honest official-metric evidence (+0.0018) and a principled difference from slot 1.
Bug fixed 2026-09-25: write_submission_csv did not clamp coordinates, so the official validator rejected every CSV
("node fields cannot be negative", ~1,880 nodes rounded to -1); the notebook's own writer clamps with max(0, .).
