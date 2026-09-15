# Manuscript-to-code terminology

Several code identifiers predate the manuscript wording and do not match it. They are retained rather than renamed because the same strings appear in the published RKpC Benchmark record — as directory names, job identifiers, manifest column values, and frozen SHA-256 hash keys — so renaming them would break the link between this repository and the deposited data. This table is the authoritative mapping; read it before mapping a result in the paper onto the code that produced it.

## The context and fusion-position study

The manuscript has no "injection" terminology. Everything below is the study the manuscript reports as tissue context and fusion position.

| Manuscript | Code |
| --- | --- |
| Tissue-context / fusion-position study | package `rat_kp_fusion`, output tree `injection_study/`, config `configs/INJECTION_STUDY_CONFIG_V1.json` |
| (same) | job identifiers prefixed `inject__`, aggregate outputs under `results/aggregate/injection/` |
| (same) | entry points `prepare_injection_study.py`, `run_injection_experiments.py`, `analyze_injection_results.py`, `deep_analyze_injection_results.py`, `run_injection_smoke.py`, `run_injection_gpu_preflight.py` |
| Fusion position (early / late) | field `injection_position`, values `early` / `late` |
| Observed-tissue evaluation | `stage = "part1"` |
| Leave-one-tissue-out (LOTO) | `stage = "loto"` |
| Exploratory analyses | `deep_dive/` output directory |

## Split schemes

| Manuscript | Code |
| --- | --- |
| Parent-group split | `split_type = "parent_group"` in the fusion and control plans; `split_type = "random"` in the core (v2) job manifests; split files named `random_seed<N>.csv` |
| Scaffold split | `split_type = "scaffold"`; split files named `scaffold_seed<N>.csv` |

**The `random` stem does not mean a random split.** Both schemes assign whole parent groups and neither ever divides a parent group across partitions (`prepare_data.split_parent_groups`, `prepare_data.scaffold_parent_split`; runtime leakage checks in `validate_setup.py`). Resolve the stem through `SPLIT_FILE_STEM` / `MANUSCRIPT_SPLIT_SCHEME` in `rat_kp_core/paths.py` rather than writing the literal.

## Context conditions

| Manuscript | Code |
| --- | --- |
| Structure only | `structure_only` |
| One-hot, early / late | `onehot_early` / `onehot_late` (context type `tissue_onehot`) |
| Physiology-based, early / late | `physiology_early` / `physiology_late` (context type `physiology`) |
| Molecule + additive tissue intercept | `additive_tissue_intercept`; package `rat_kp_controls`, output tree `additional_context_analyses/`, config `ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json` |

## Architectures

| Manuscript | Code |
| --- | --- |
| GCN, GINE, D-MPNN, AttentiveFP | `gcn`, `gine`, `d_mpnn`, `attentive_fp` |

`random_forest`, `xgboost`, and `linear_svr` appear in `rat_kp_core` and in `FIXED_MODEL_CONFIG_V2.json` but are **not reported in the manuscript**. They belong to the core (v2) stage, which the fusion stage depends on only for its reused D-MPNN and AttentiveFP predictions.

## Output propagation and literature panels

| Manuscript | Code |
| --- | --- |
| Tissue distribution-volume index (DVI), Vss propagation | package `rat_kp_dvi`, output tree `vss_output_propagation/`, config `VSS_OUTPUT_PROPAGATION_CONFIG_V1.json` |
| PT / RR matched direct panels | `prepare_matched_panels.py` → `results/local_matched_panels/`; `evaluate_manuscript_benchmarks.py` → `results/manuscript_benchmarks/`; distributed as `separate_literature_benchmarks/` |

## "legacy"

`legacy` throughout `rat_kp_fusion` (`legacy_reuse`, `LEGACY_CONTEXT_MAP`, `legacy_job_id`, `legacy_fixed_config_sha256`, and `load_legacy_config` in the test suite) refers to the **core (v2) stage of this same study**, not to superseded or deprecated work. The fusion stage reuses completed core D-MPNN and AttentiveFP runs for the `structure_only`, `onehot_late`, and `physiology_late` conditions instead of retraining them, and `legacy` marks those reused rows.

## Frozen-configuration path strings

`configs/ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json` records `data/processed/rat_kp_long.csv` and `manuscript/INJECTION_STUDY_CONFIG_V1.json`. These name the working tree used during execution; the public equivalents are `data/inputs/rat_kp_long.csv` and `configs/INJECTION_STUDY_CONFIG_V1.json`. The configurations are SHA-256 frozen and are never edited. See `docs/reproducibility_spec.md`.
