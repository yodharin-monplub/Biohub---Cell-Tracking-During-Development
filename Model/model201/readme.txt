MODEL201 - ULTRA-GENTLE FINE-TUNE OF THE PUBLIC PRIMARY (lr 5e-6, 8 epochs)

Third point on the dose-response curve found with the model199 local check (nodes on the hard visible movie
44b6_0b24845f; public = 20,707):
  model197 (lr 2e-5, 48 ep)  13,497
  model193 (lr 2e-5, 24 ep)  14,758
  model196 (lr 1e-5, 12 ep)  20,590   <- keeps the public recall, submitted 2026-09-23
The less we move the weights, the less recall is lost on hard movies. model201 moves them ~3x less again
(lr 5e-6 x 8 epochs) to test whether an even smaller update is safer still, or simply does nothing.

Run:    powershell -File Model\model201\run_train.ps1  (detached via WMI; waits for the GPU)
Work:   C:\biohub_data\work\model201
Output: Model\model201\weights\model201_ft_primary\
Then:   model199 local check before any Kaggle push.

2026-09-23 01:35: local check @0.965: total 126,318 nodes, hard movie 23,757 (public 20,707, +15%). Less training gave MORE detections, not fewer - the curve is not monotone. Held: model196 (on the public count) is the submitted candidate.
