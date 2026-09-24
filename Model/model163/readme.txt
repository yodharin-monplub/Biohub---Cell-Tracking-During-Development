MODEL163 - SOURCE-ONLY LOW-EDGE-FLOOR ELIGIBILITY AUDIT

Difficulty 5/5. Status: read-only source-side audit completed; its
p0.35 recommendation was tested in model164 and did not improve score.

Hypothesis: model161's calibrated ranking still excludes useful
top-five links at the fixed p0.40 ILP candidate floor. Audit only the
eight source6bba movies held out from calibrator fitting, using the
model161 source-fit (24-movie) calibrator and exact per-movie rank mapping.
Count sparse-label-testable positive links and total solver candidates
retained at floors0.30,0.35,0.40. The audit neither solves graphs nor
touches outer44b6 labels or model161 output. It cannot establish a
tracking score, and source movies were seen by the backbone during
training. Only consider a later target run if source evidence is
material and the remaining 10-hour runtime allowance permits it.
If a threshold is selected, the separate model161 full-source refit
would be used at target inference; the source-dev threshold audit must
not itself refit on its eight evaluation movies.

Run: .venv-gpu/bin/python -u model163/audit_source_floor.py

SOURCE-DEV AUDIT RESULT (2026-09-14)
All eight source-calibrator-dev movies completed in18.583 seconds.
At p0.40 the rankmapped graphs retained244,310 total solver edges,
including7,175 testable true and1,629 testable false links. At p0.35
they retained257,663 total edges (+5.47%),7,256 true (+81,1.13%)
and1,883 false (+254). At p0.30 they retained273,052 total edges,
7,325 true (+150) and2,238 false (+609). Choose0.35 as the bounded
next graph-score test: it recovers testable true links with a smaller
candidate-set increase than0.30. These are eligibility counts, NOT
solved/final graph scores; no target labels were used. See
source_floor_audit.json.
