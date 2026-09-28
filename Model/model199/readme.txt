MODEL199 - DETECTION-THRESHOLD RECALIBRATION FOR THE FINE-TUNED PRIMARY (model193)

Evidence (dummy-test commit outputs vs public LB):
  model167 public checkpoints        241k rows / 122.8k nodes  -> 0.947
  model193 fine-tuned primary        227k rows / 115.7k nodes  -> 0.938  (-6% nodes)
  model190 'primary' (from scratch)  304k rows                 -> 0.920  (+26%)
  model190 'both' (from scratch)     292k rows                 -> 0.900  (+21%)
The pipeline's DET_THRESHOLD (0.965) and guards were tuned to the public checkpoints; changing the checkpoint shifts
the detection count and the score drops with the size of the shift. Hypothesis: model193 with a threshold that
restores the public node count keeps the fine-tune's gain without the recall loss.

Step 1 (local, no Kaggle GPU): sweep.ps1 runs the frozen notebook locally (make_local_variant.py: threshold from env,
guard patched to match, optional primary swap; validator off) on the 4 visible test movies:
public@0.965 (reference) and model193 @ 0.965 / 0.95 / 0.93 / 0.90. Results in C:\biohub_data\work\model199\<tag>\summary.json.
Step 2: pick the threshold whose node count matches the public reference, build a Kaggle notebook (model193 build +
threshold/guard patch), dummy-data commit, submit.

2026-09-22 22:25 sweep results (4 visible test movies, nodes total | hard movie 44b6_0b24845f):
  public@0.965 122,749 | 20,707;  m193@0.965 115,711 | 14,758;  m193@0.95 117,300 | 15,780;  m193@0.93 118,080 | 16,117
  The whole deficit is one hard movie (-29%); other movies are within ~1-3% and pass the public counts at 0.93.
  A global threshold only partly recovers the hard movie, so threshold recalibration alone is unlikely to rescue model193.
  sweep2.ps1 queued: model196 (gentle) and model197 (longer) at 0.965 to see whether a gentler fine-tune keeps the hard-movie recall.

2026-09-23 01:35 full local dose-response (nodes on hard movie 44b6_0b24845f / total, 4 visible test movies @0.965; public = 20,707 / 122,749):
  model201 lr5e-6 x 8ep   23,757 / 126,318  (+15%)
  model196 lr1e-5 x12ep   20,590 / 123,543  (-0.6%)  <- submitted 2026-09-23
  model193 lr2e-5 x24ep   14,758 / 115,711  (-29%)   LB 0.938
  model197 lr2e-5 x48ep   13,497 / 114,037  (-35%)
  pair m196+m198          22,636 / 125,038  (+9%)
  NOT monotone: a short fine-tune raises detection counts, longer training collapses them; model196 sits on the public count.

2026-09-23 10:50: public primary + model198 as SECONDARY only: total 123,936, hard movie 22,701 (+9.6% vs public). Same drift pattern that cost 0.005-0.009 every time -> NOT submitted. Checkpoint-replacement line closed.
