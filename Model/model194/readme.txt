MODEL194 - FINE-TUNE THE PUBLIC SECONDARY CHECKPOINT (companion to model193)

Same recipe as model193 (all 199 train movies, lr 2e-5, 24 epochs x 250 it x batch 2) but warm-started from the
public secondary seed (secondary-seed split_0, sha256 9bac2fa0...), seed 20260924. Trained while model193 is
being scored, so that if model193 beats 0.947 on the LB a fine-tuned PAIR (model193 primary + model194 secondary)
can be submitted next. If model193 does not beat 0.947, this is not used.

Run:    powershell -File Model\model194\run_train.ps1   (launch detached via WMI Win32_Process.Create)
Work:   C:\biohub_data\work\model194
Output: Model\model194\weights\model194_ft_secondary\

2026-09-22 10:25: trained (sha256 509c2222b1581085febe7d24c0df5c99cedb3cb5ba9b67019dd449fea42f5b5e); uploaded privately as krittanutsomtuas/biohub-model194-ft-secondary-weights; used by model195.
