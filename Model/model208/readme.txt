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
