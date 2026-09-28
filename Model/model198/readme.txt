MODEL198 - GENTLE FINE-TUNE OF THE PUBLIC SECONDARY (lr 1e-5, 12 epochs)

Partner for model196 (gentle primary), the way model194 partners model193: warm start from the public secondary seed
(sha256 9bac2fa0...), all 199 train movies, lr 1e-5, 12 epochs x 250 it x batch 2, seed 20260927. Queued behind
model197 on the single GPU. Used only if the LB shows the gentle fine-tune is the right strength (then a
model196 + model198 pair can be submitted).

Run:    powershell -File Model\model198\run_train.ps1   (launch detached via WMI)
Work:   C:\biohub_data\work\model198
Output: Model\model198\weights\model198_ft_secondary\

2026-09-23 00:05: local check of the gentle PAIR (model196 primary + model198 secondary) @0.965: total 125,038 nodes (+1.9% vs public 122,749), hard movie 22,636 (+9.3% vs 20,707). More detections than the public baseline - could be recall or false positives; held until model196's LB score arrives.
