MODEL160 - HELD-OUT EDGE GEOMETRY AUDIT

Status: diagnostic only; no candidate, training, GPU inference, Kaggle run,
or submission. Difficulty 5/5 for the 0.97 cross-embryo CV goal.

Question: Is the large held-out association loss of model159 explained by
GT links longer than its fixed 12-voxel (~19.5 um) candidate-edge gate,
or by coherent whole-frame shifts? Analyze all 71 held-out 44b6 movies,
using immutable model159 official scores and GT geometry. The raw GT may
be inspected only for this retrospective diagnosis. Do not tune or publish
a purported CV improvement from these same held-out labels.

Run: .venv-gpu/bin/python model160/audit_geometry.py
Output: model160/geometry_audit.json (write-once)

The candidate cap equivalence follows the exporter downsample [1,4,4]
and original scale [1.625,0.40625,0.40625] um/voxel, yielding an
approximately isotropic 1.625 um model grid. This audit measures only
consecutive GT edges; nonconsecutive annotation gaps are counted
separately. A high fraction would justify a separately precommitted
motion-aware inference experiment; a low fraction would reject this
particular explanation without changing the frozen model1 control.

The second read-only audit, audit_candidate_oracle.py, matches the saved
overcomplete model159 candidate graph to GT at the organizer's 7-um
node radius. It partitions each GT edge into missing candidate endpoint,
no exported top-five link, exported link below ILP p0.40, eligible link
dropped by ILP, and ILP-selected link. These are candidate-stage oracle
counts, not exact attribution of the final organizer edge TP/FN, because
matching on an overcomplete graph can differ from final-graph matching.
Run: .venv-gpu/bin/python model160/audit_candidate_oracle.py
Output: model160/candidate_oracle.json (write-once)

RESULTS (2026-09-14)
Both read-only audits completed on all 71 outer44b6 movies. The geometry
audit found 19,826/19,826 consecutive GT edges at or below the model's
19.5-um candidate cap; the longest was 18.692 um. There were no
nonconsecutive GT edges and no movie with a multi-edge coherent frame
shift over that cap. Thus increasing the max candidate distance would
not explain the current held-out missed GT links. This does not deny
smaller frame motion or photometric domain shift.

The candidate/ILP audit partitioned all 19,826 GT edges using 7-um
matching on the OVERCOMPLETE graph:
  unmatched candidate endpoint          852 (4.3%)
  endpoints matched, no exported link   217 (1.1%)
  exported link below ILP p0.40       1,843 (9.3%)
  eligible link dropped by ILP        1,049 (5.3%)
  link selected by ILP               15,865 (80.0%)
The official FINAL model159 metric reports 17,401 edge TP and 2,425 FN.
The candidate-stage matched node correspondence is not identical to the
final-graph scorer's match and notebook repairs add links, so these
stage counts must NOT be subtracted from official FN as exact causes.
They do show a substantial association-confidence/selection frontier
relative to a much smaller no-exported-link category. Both audits
used held-out labels for diagnosis; choosing a new inference threshold
from them would make the same fold a development set, not independent
CV confirmation. No new model was run or score improved by this audit.

FOCUS-3D was also checked as a possible independent detector. Its
upstream code is BSD-3-Clause and the model card says Apache-2.0 for
weights, but accessing the pretrained weights requires an authenticated
user to accept contact-sharing conditions. No checkpoint was downloaded,
mirrored, or used. Sources:
https://github.com/yu-lab-vt/FOCUS-3D
https://huggingface.co/Qinghua-thu/FOCUS-3D
