MODEL104 RE-UPLOAD AND AUTHORIZED COMPETITION SUBMISSION

User explicitly requested: "upload model104 again and submit the run".
Re-upload accepted2026-09-10 as version1, new kernel ID133826128, private slug
yodharinmonplub/biohub-model104-protected-motion-links, explicit NvidiaTeslaT4.
The exact Vast-verified deployment notebook was uploaded; no model changes.

The prior deleted notebook had a different ID133812292. Its rejected direct
submission attempt remains separately recorded in hidden_submission_attempt.json.
Never use that attempt as proof this new notebook was submitted.

monitor_reupload.py checks every600seconds without alarms. It verifies the
kernel ID/version/code and accelerator, waits for the committed run, downloads
and validates submission.csv, checks checkpoint integrity and model104 activity,
then makes one authorized code-competition submission of version1. The attempt
is recorded before the API call to prevent accidental duplicate submissions.
It then monitors the returned submission ID for scoring completion.

Live state: reupload1/status.json. Evidence: remote_verification.json,
validation.json, submission_attempt.json, submission_receipt.json, scoring_result.json.
Only submission_receipt.json with a positive ID confirms successful submission.
The normal committed run uses weekly GPU quota; the hidden scoring rerun is
separate. This supervisor never pushes or creates additional notebook sessions.
