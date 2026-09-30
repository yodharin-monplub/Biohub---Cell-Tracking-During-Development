MODEL209 - 0.947 PIPELINE + OWN DIVISION DETECTOR PROPOSING EXTRA DIVISIONS (Kaggle submission)

Difficulty 4/5 (the gain lives in a tiny, sparse part of the metric; validation has few events).

Why: the notebook's scorer is  adjusted edge Jaccard + 0.1 x division Jaccard. The 0.947 pipeline recovers only
3 of 12 annotated divisions on its 8 validator movies. One more correct division is worth ~+0.0077, one more false
one ~-0.0016, and divisions on unannotated cells are not counted - so confident extra divisions pay. The missed
divisions are cut by the safe-division step's mutual-NN / divergence filters (and the DeepCenter veto) before any
ranking, so vetoing or re-ranking changed nothing (Model\model208).

What: Model\model207 trained our own division-event detector on the 151 annotated divisions (movie-grouped 4-fold
CV AUC 0.907; the public DivNet artifact reaches 0.562). model209 lets it override those filters for parents it
scores above 0.40, up to 30 extra divisions per movie, queued after every division the unchanged pipeline makes.

Local evidence (8 validator movies, identical predictions, each train movie scored with the fold that never saw it):
  unchanged pipeline         0.949047   divisions 3 found / 1 false / 9 missed
  propose 0.80 / 3           0.949026   3 / 1 / 9
  propose 0.60 / 10          0.946281   3 / 3 / 9
  propose 0.40 / 30          0.955383   5 / 5 / 7     <- submitted
  propose 0.30 / 50          0.955378   5 / 5 / 7
Only 12 annotated divisions, so a 40-movie confirmation (Model\model208\run_confirm40.ps1) runs in parallel.

Build:   Other\.venv\Scripts\python.exe Model\model209\build.py 0.40 30
Dataset: krittanutsomtuas/biohub-model209-divgate (div_gate.py + model207 fold0-3.pt), private
Kernel:  krittanutsomtuas/biohub-model209-division-propose (commit = dummy data only, validator off at commit)
Monitor: set $env:KAGGLE_API_TOKEN from Other/.env, then Other\.venv\Scripts\python.exe Model\model209\kaggle\monitor_once.py [--submit]
         The monitor refuses to submit unless model209_gate.json proves the detector loaded (mode propose, 4 folds).

2026-09-24 17:25: commit v1 validated on the dummy test data: 241,461 rows, 122,831 nodes (+0.07% vs public 122,749), 218 divisions (public 124); gate receipt: mode propose, 4 folds, 0.40/30. Submitted.

2026-09-25 01:50 - 40-movie local confirmation (60 annotated divisions): baseline 0.916723 (13/28/47) -> propose 0.40/30 0.918214 (18/50/42), +0.0015. Direction holds; size much smaller than the 8-movie +0.0063. Added-division precision ~19% vs ~17% break-even.

2026-09-25 ~03:45: PUBLIC LB 0.945 - below 0.947. Adding confident extra divisions (P>0.40, <=30/movie) hurt on the hidden test despite +0.0015 on 40 local movies.
