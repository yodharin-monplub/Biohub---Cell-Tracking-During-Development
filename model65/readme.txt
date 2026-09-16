MODEL 65 — LEAVE-ONE-MOVIE-OUT CALIBRATED TOP-K ASSOCIATION

Status
------
Complete strict graph-level OOF evaluation. This is not a submission.

Hypothesis
----------
Model64's LOO candidate audit improves AUC from 0.99417 to 0.99460 and
top-parent accuracy by six annotated targets. Reweight the frozen top-five
edges using only each held-out movie's complementary labels and evaluate the
actual ILP graph, retaining the existing raw p=0.40 candidate gate.

Protocol
--------
For each of the 20 broad movies, fit a ridge logistic calibrator on labelled
candidates from the other 19. Features are raw probability/logit, distance,
target/source ranks and margins, candidate counts, and one predeclared
probability-distance interaction. Update only the in-memory score used by the
ILP; nodes, raw candidate floor, all ILP penalties, and division behavior are
unchanged.

Promotion gate
--------------
Require an exact OOF graph-score improvement over model48 with family support,
runtime within the baseline envelope, and no selection of calibration settings
using the held-out graph score.

Results
-------
The exact strict-LOO graph score is 0.9047669533, +0.0005735012 over model48.
The solve completes in 312.5 seconds across all 20 movies. Both family means
improve, but only 6/20 movie scores improve and the pooled gain is concentrated
in two 44b6 movies. It is therefore a useful OOF signal, not a standalone
production promotion.
