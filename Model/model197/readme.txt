MODEL197 - LONGER FINE-TUNE: model193 CONTINUED FOR 24 MORE EPOCHS (48 total at lr 2e-5)

Warm start from model193 (public primary fine-tuned 24 ep, sha256 710a6379...), 24 more epochs x 250 it x batch 2,
lr 2e-5, seed 20260926, all 199 train movies. The "more of it" arm: if model193 beats 0.947, this is the next
candidate; model196 (gentler) is the arm for the opposite outcome. Starts automatically when model196 finishes
(single 6 GB GPU).

Run:    powershell -File Model\model197\run_train.ps1   (launch detached via WMI)
Work:   C:\biohub_data\work\model197
Output: Model\model197\weights\model197_ft_primary\

2026-09-22 19:05: trained (sha256 975523bec93b6878a2d579b970d6116a05da61d4c249a9394df2187d265cf671); uploaded krittanutsomtuas/biohub-model197-ft-primary-weights; notebook built + monitor pinned. Push only if model193 beats 0.947.

2026-09-22 23:30: local check @0.965: total 114,037 nodes, hard movie 44b6_0b24845f 13,497 (public 20,707; model193 14,758; model196 20,590). More fine-tuning = more hard-movie recall loss. NOT pushed.
