# Reproducibility specification

This document records the executable specification used for the tissue-context injection study. It supplements `environment.yml`, the frozen JSON configurations, and the per-job `run.json` records.

## Input and output locations

Every path is resolved from the repository root, so scripts behave identically regardless of the working directory they are launched from. The constants are defined once and imported; no pipeline restates a directory literal.

| Artifact | Location | Defined in |
| --- | --- | --- |
| Request-only processed datasets | `data/inputs/` | `rat_kp_core/paths.py` |
| Primary splits, LOTO splits | `data/inputs/splits/`, `data/inputs/splits/loto/` | `rat_kp_core/paths.py` |
| Sensitivity splits | `data/inputs/splits_sensitivity/{ad_excluded,condition_specific}/` | `rat_kp_core/paths.py` |
| Core (v2) job results | `results/experiments/` | `rat_kp_core/paths.py` |
| Injection-study results | `injection_study/results/` | `rat_kp_fusion/paths.py` |
| Additional-context results | `additional_context_analyses/results/` | `rat_kp_controls/paths.py` |
| DVI propagation results | `vss_output_propagation/results/` | `rat_kp_dvi/paths.py` |
| Local PT/RR matched panels | `results/local_matched_panels/` | `rat_kp_dvi/paths.py` |
| Distributed aggregate outputs | `results/aggregate/` | `scripts/build_figures.py` |

`data/inputs/` and every results directory are excluded by `.gitignore`. The splits are written by `prepare_data.write_splits` and `prepare_data.write_derived_splits`, which import the same constants the readers use.

## Pipeline execution order

The four pipelines are not independent. `rat_kp_fusion` reuses completed `rat_kp_core` jobs for the D-MPNN and AttentiveFP runs in the `structure_only`, `onehot_late`, and `physiology_late` conditions (`rat_kp_fusion.jobs._execution_mode` marks these `legacy_reuse`), so the core stage must complete before the injection stage is planned.

1. `prepare_data.py` — build the processed datasets, the frozen splits, and the tabular feature cache.
2. `prepare_experiment.py` — write the frozen primary and robustness job manifests into `manifests/`.
3. `run_preflight.py --device cpu` and `--device gpu` — 15 reduced integration jobs each; writes `results/preflight/`.
4. `validate_setup.py` — fail-closed pre-experiment audit; writes `manifests/pre_experiment_validation_v2.json`. It requires that no completed job exists yet, so run it before any full job and before unpacking a Zenodo `results/experiments/` tree.
5. `run_full_study.py` — core (v2) jobs for `random_forest`, `xgboost`, `linear_svr`, `d_mpnn`, `attentive_fp`, then `analyze_results.py`. These runs supply the reused predictions consumed in step 6; the three tabular baselines are not reported in the manuscript.
6. `prepare_injection_study.py` → `run_injection_experiments.py` → `analyze_injection_results.py` → `deep_analyze_injection_results.py` — the four-GNN context and fusion-position study reported in the manuscript.
7. `prepare_additional_context_analyses.py` → `run_additional_context_analyses.py` → `analyze_additional_context_analyses.py` — the additive tissue-intercept control and preprocessing robustness analyses.
8. `prepare_matched_panels.py`, then `run_vss_output_propagation.py` → `analyze_vss_output_propagation.py` — matched PT/RR panels and DVI output propagation.
9. `scripts/build_figures.py`, then `scripts/audit_submission_figures.py`.

## Analysis outputs and the distributed aggregate layout

Analyses write beside their own pipeline; the Zenodo record collects them under `results/aggregate/`. The two layouts use different directory names for the same files, so a locally regenerated output must be compared against the aggregate path listed here rather than against a same-named path.

| Producer | Local output | Distributed aggregate path |
| --- | --- | --- |
| `analyze_injection_results.py` | `injection_study/results/analysis/*.csv` | `results/aggregate/injection/` |
| `deep_analyze_injection_results.py` | `injection_study/results/analysis/deep_dive/*.csv` | `results/aggregate/injection/deep_dive/` |
| `analyze_additional_context_analyses.py` | `additional_context_analyses/results/analysis/*.csv` | `results/aggregate/additional_context/` |
| `evaluate_manuscript_benchmarks.py` | `results/manuscript_benchmarks/*.csv` | `results/aggregate/injection/separate_literature_benchmarks/` |

`analyze_vss_output_propagation.py` writes `vss_output_propagation/results/analysis/*.csv`. The figure builder does not read those files; consult the RKpC Benchmark record for the directory they are deposited under.

`scripts/build_figures.py` reads only the aggregate layout. The files it requires are:

- `injection/part1_context_effects.csv`, `injection/part1_position_effects.csv`, `injection/loto_inference.csv`
- `injection/deep_dive/primary_performance_summary.csv`, `injection/deep_dive/loto_context_extrapolation.csv`
- `injection/separate_literature_benchmarks/direct_panel_metrics.csv`
- `injection/separate_literature_benchmarks/chemical_space/chemical_space_compounds.csv` and `chemical_space_summary.json` (these require request-only structure-level inputs)
- `additional_context/all_additional_job_metrics.csv`, `additional_context/loto_raw_inference.csv`, `additional_context/loto_preprocessing_comparison.csv`

## Frozen-configuration path strings

`configs/*.json` are SHA-256 frozen and recorded in `configs/config_hashes.csv`, `MANIFEST_SHA256.csv`, and the archived run manifests, so their contents are never edited. Two path strings inside them refer to the working tree used during execution and do not exist in the public layout:

| Frozen string | Public equivalent |
| --- | --- |
| `data/processed/rat_kp_long.csv` (`ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json`) | `data/inputs/rat_kp_long.csv` |
| `manuscript/INJECTION_STUDY_CONFIG_V1.json` (`ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json`) | `configs/INJECTION_STUDY_CONFIG_V1.json` |

The code never reads these strings as paths; they are provenance records only.

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

## Resampling seeds

Every interval in the study comes from a seeded bootstrap, so each reported interval is exactly reproducible. The seeds differ per analysis because each was fixed when that analysis was frozen; they are recorded here because a reader reproducing an interval needs the seed, not only the replicate count.

| Analysis | Seed | Defined in |
| --- | --- | --- |
| Core (v2) observed-tissue and LOTO inference | 20260619 | `rat_kp_core/statistics.py` |
| Tissue-context / fusion-position study | 20260803 | `rat_kp_fusion/statistics.py` |
| Literature benchmarks and fold accuracy | 20260805 | `evaluate_manuscript_benchmarks.py`, `analyze_fold_accuracy.py` |
| DVI output propagation | 20260828 | `configs/VSS_OUTPUT_PROPAGATION_CONFIG_V1.json` |

All use 10,000 replicates. Where one analysis reports many intervals, a per-quantity offset is added to the seed so each draws its own resampling indices while staying reproducible; where a single interval is reported, the seed is used directly.

`rat_kp_core.statistics` and `rat_kp_fusion.statistics` each define their own `exact_sign_flip_pvalue` and `benjamini_hochberg`. They are kept separate rather than shared: the two stages were frozen at different times, and the fusion copy additionally validates its input. Changing either would alter a recorded analysis, so they are deliberately not deduplicated.

The sign-flip test enumerates all `2**n` sign assignments and is therefore exact rather than sampled. It is capped at twenty paired differences; the real families are ten split seeds or eleven held-out tissues, so the cap cannot be reached by a legitimate run and exists only to turn an accidental misuse into an error rather than an apparent hang.

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
