MODEL172 - RADIUS/DIVISION TRADEOFF INSTRUMENTATION

Difficulty 5/5. Status: complete diagnostic; no production change.

Purpose: explain the single extra division false positive introduced on
6bba_062c8d37 by the model1715.15-5.35 motion-radius plateau. The exact
model167 pipeline, weights and predictions are frozen. Only two postprocess
controls are replayed:5.15 (new plateau) and5.45 (same output as5.5).

Instrumentation records label-free DeepCenter repair scores, predicted
division geometry, safe-division provenance, and validator node matching for
each movie/config. It does not change acceptance logic or claim a score. Ground
truth is used only after prediction to classify the observed tradeoff and must
not be used to create a movie-ID rule. Any subsequent safeguard must be
predeclared from a biological, label-free feature and rechecked across all
movies. This is tuning analysis, not honest CV or evidence of0.97.

RESULT (2026-09-15)
The frozen replay completed and reproduced5.15 at0.9523628978 and5.45 at
0.9511066278. On6bba_062c8d37,5.15 correctly rematched a parent to its
ground-truth continuation but the existing safe-division stage then attached
an unmatched second child. This made a formerly unmatchable predicted branch
count as the one additional official division false positive. The affected
parent's existing edge probability was0.98835 and the candidate DeepCenter
score was0.44211; the preserved true division scored0.57693.

Across all eight movies, the instrumented5.15 output contained three predicted
divisions whose parent matched a non-division ground-truth node, with candidate
DeepCenter scores approximately0.282,0.427 and0.442. The two exactly matched
true divisions scored0.510 and0.577. This motivated model173's full threshold
replay, but the sparse diagnostic separation did not generalize to all official
division events. Raw per-movie records are under output/model172_audit.
