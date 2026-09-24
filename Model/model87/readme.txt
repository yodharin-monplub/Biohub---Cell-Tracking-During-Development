MODEL 87 — PRIVATE KAGGLE CHECKPOINT DATASET

Status
------
Prepared for the explicitly authorized Kaggle upload.

Purpose
-------
Package the two selected half-step checkpoints as one private Kaggle dataset so
the model85 and model86 code submissions can rerun inference without internet.

Files
-----
- fold3_edge_predictor_best.pth: model86 higher-upside checkpoint.
- fold4_edge_predictor_best.pth: model85 conservative checkpoint.
- config.json: architecture and preprocessing configuration.
- ARTIFACT_MANIFEST.json: immutable source/checksum provenance.
- dataset-metadata.json: private Kaggle dataset metadata.

The checkpoint files are copied mechanically from model82 and verified before
upload. The dataset is private and is not a competition submission by itself.

