# Reproducibility specification

This document records the executable specification used for the tissue-context injection study. It supplements `environment.yml`, the frozen JSON configurations, and the per-job `run.json` records.

## Recorded runtime

- Python 3.11.15
- PyTorch 2.11.0+cu128 (CUDA 12.8 build)
- PyTorch Geometric 2.7.0
- Lightning 2.6.1
- Chemprop 2.2.3
- RDKit 2026.03.1
- NVIDIA GeForce RTX 4090; one GPU per neural job

The selected RTX 4090 device UUID is retained in the original launch log. A device UUID is intentionally not required by the public code because an equivalent CUDA-capable device may be used.

## Molecular graph features

All four GNN families use the Chemprop v2 `SimpleMoleculeMolGraphFeaturizer` defaults.

- Atom vector: 72 dimensions covering atomic number, total degree, formal charge, chiral tag, total hydrogen count, hybridization, aromaticity, and scaled atomic mass. Categorical groups contain an unknown-value slot.
- Bond vector: 14 dimensions covering a null-bond indicator, single/double/triple/aromatic bond type, conjugation, ring membership, and bond stereochemistry.
- Early context is concatenated to every atom vector before the first message-passing operation. For D-MPNN it is supplied as the extra atom feature used in directed-bond message initialization.
- Late context is concatenated to the pooled molecular representation immediately before the prediction head.

## Seeds and data partitions

- Parent-group and scaffold evaluations use split seeds 0 through 9.
- For those evaluations, the training seed equals the split seed.
- LOTO evaluation uses training seeds 0 through 4 for every held-out tissue.
- Parent-group, scaffold, and LOTO assignments are stored as frozen split files. SHA-256 hashes are captured in the pre-experiment validation manifest.
- Runtime checks reject parent-group leakage and identity mismatches before training.

## Shared neural training controls

- Target: `log10(Kp)`
- Loss: mean squared error
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.00001
- Batch size: 64
- Maximum epochs: 100
- Early stopping: validation loss, patience 15, minimum improvement 0.0001
- Gradient clipping: 5.0
- Deterministic Lightning execution: enabled
- Data-loader workers: 0
- No condition-specific hyperparameter optimization

The lowest-validation-loss state is retained in memory and restored before test prediction. Full hyperparameters are defined in `configs/INJECTION_STUDY_CONFIG_V1.json` and `configs/FIXED_MODEL_CONFIG_V2.json`.

## Audit trail

The pre-experiment validation record contains software versions, source and split hashes, leakage checks, smoke-test results, and trainable parameter counts. Every completed job stores its dataset hash, configuration hash, best epoch, validation loss, parameter count, test metrics, runtime, and prediction file. Prediction rows preserve the job, split, seed, source-row, parent-group, identity-unit, SMILES, tissue, observed-target, and predicted-target fields.
