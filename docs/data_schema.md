# Processed-data schema

## Primary and sensitivity datasets

The files `rat_kp_long.csv`, `rat_kp_long_ad_sensitivity.csv`, and `rat_kp_long_condition_specific.csv` use the same analysis-facing schema.

Required training columns:

| Column | Meaning |
| --- | --- |
| `row_index` | Frozen record identifier |
| `parent_group_id` | Leakage-control grouping for the parent chemical entity |
| `identity_unit_id` | Frozen standardized molecular-identity unit |
| `Drug` | Harmonized compound label |
| `SMILES` | Standardized molecular structure |
| `Species` | Species label; the present study uses rat records |
| `Tissue` | Lower-case tissue label |
| `Kp` | Positive harmonized tissue-to-plasma partition coefficient |
| `log10Kp` | Base-10 logarithm of `Kp` used as the prediction target |

Additional audit columns may be present in the request package. Training code ignores columns that are not required by the frozen contract.

## Split files

Each split file must contain:

| Column | Meaning |
| --- | --- |
| `row_index` | Record identifier matching the processed dataset |
| `parent_group_id` | Parent-group identifier copied from the processed dataset |
| `identity_unit_id` | Identity-unit identifier copied from the processed dataset |
| `SMILES` | Standardized molecular structure copied from the processed dataset |
| `Tissue` | Tissue label copied from the processed dataset |
| `split` | `train`, `val`, or `test` |

## Tissue-physiology file

`rat_tissue_physiology.csv` contains one row for each of the 11 tissues and the following columns:

- `tissue`
- `neutral_lipid_fraction`
- `neutral_phospholipid_fraction`
- `extracellular_water_fraction`
- `intracellular_water_fraction`
- `acidic_phospholipid_mg_g`

The study-specific values are part of the request-only processed-data package and are not stored in the public code repository.

## Optional local manuscript-benchmark reconstruction

The RKpC Benchmark contains aggregate manuscript benchmark results, but it does not redistribute record-level PT/RR values. To independently recalculate the literature comparison, an authorised user must reconstruct `pt_benchmark_processed.csv` and `rr_benchmark_processed.csv` locally from the cited original articles and errata with the following schema:

- `workbook_excel_row` (stable processed-record identifier)
- `workbook_drug`
- `Tissue`
- `canonical_smiles`
- `benchmark_method`
- `baseline_pred_kp`
- `paper_exp_kp`
- `y_true_log10`
- `baseline_pred_log10`

`direct_panel_record_predictions.csv` is likewise a non-distributed local reconstruction. It adds the method-specific held-out GNN prediction, split type, selected model and condition, and OOF split-seed counts. `internal_gnn_seed_metrics.csv` is distributable and supplies the frozen seed-level internal evaluation used for the three-domain descriptive summary.

The expected PT/RR source locations and transformation rules are documented in this repository and the RKpC Benchmark. The public code does not grant rights to, extract, or redistribute the source-publication values.
