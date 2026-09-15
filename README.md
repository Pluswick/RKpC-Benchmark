# RatKpContext

RatKpContext is the software repository supporting the manuscript **Dissecting Tissue-Context Effects in Graph Neural Networks for Rat Kp Prediction: Additive Shifts, Fusion Position, and Extrapolation**.

## Manuscript-to-code terminology

Several identifiers predate the manuscript wording. Most importantly, the study the manuscript reports as **tissue context and fusion position** is named `injection` throughout the code (package `rat_kp_fusion`, directory `injection_study/`, job prefix `inject__`), and the **parent-group** split files are named `random_seed<N>.csv` even though no split ever divides a parent group. These names are retained because they appear in the published RKpC Benchmark record and in frozen hash keys, so renaming them would break the link to the deposited data. See **[`docs/terminology.md`](docs/terminology.md)** for the full mapping; read it before matching a result in the paper to the code that produced it.

## Repository boundary

This repository contains code only: model definitions, training and evaluation pipelines, frozen configurations, synthetic tests, statistical analysis, DVI propagation, matched-panel construction, figure generation, and environment specifications. It does not contain the model-development dataset, record-level source links, PT/RR record-level values, real record-level predictions, frozen split assignments, or aggregate study outputs.

Publicly distributable aggregate outputs, run metadata, index-only split assignments, a Kp_Data source bibliography with aggregate contribution counts, and their hashes are provided in the [**RatKpContext Benchmark (RKpC Benchmark)**](https://doi.org/10.5281/zenodo.22671239) Zenodo record. The harmonized model-development dataset and record-level source linkage remain available from Kyeong-Ryoon Lee on reasonable request. Record-level PT/RR source transcriptions are not redistributed.

`10.5281/zenodo.22671239` is the all-versions (concept) DOI and always resolves to the latest deposited version; use it when citing the benchmark generally. The manuscript's Data availability statement cites `10.5281/zenodo.22683468`, the version DOI of the specific deposit the reported analyses were run against. Both point at the same record.

## Recorded environment

- Python 3.11.15
- PyTorch 2.11.0+cu128
- PyTorch Geometric 2.7.0
- Lightning 2.6.1
- Chemprop 2.2.3
- RDKit 2026.03.1 (spelled `2026.3.1` on PyPI, as pinned in `requirements.txt`)
- Matplotlib 3.10.9
- NVIDIA GeForce RTX 4090, CUDA 12.8

Create the recorded environment with `environment.yml`. The exact top-level package pins are also listed in `requirements.txt`.

## Synthetic verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The public tests use synthetic molecular inputs. They run a forward pass for GCN, GINE, and AttentiveFP under all five context/fusion conditions, and check that early and late fusion give the D-MPNN different trainable parameter counts. They also cover the tissue-intercept control, DVI aggregation, the statistical helpers, and value-free matched-panel construction. The D-MPNN forward pass is not exercised, because it needs the Chemprop batch machinery rather than a synthetic PyG graph.

## Data-dependent reproduction

1. Obtain the request-only processed model-development data from the corresponding author and place the processed tables (`rat_kp_long.csv`, the two sensitivity tables, and `rat_tissue_physiology.csv`) in `data/inputs/`.
2. Download the RKpC Benchmark from Zenodo and unpack it so that its index-only split assignments land in `data/inputs/splits/` (with `data/inputs/splits/loto/` and `data/inputs/splits_sensitivity/<analysis_set>/`) and its aggregate outputs land in `results/` at the repository root. Every reader resolves these locations from the repository root, so scripts may be launched from any working directory.
3. Follow `docs/data_schema.md` and `docs/reproducibility_spec.md`. The latter documents the pipeline execution order and the mapping from locally regenerated analysis outputs to the distributed `results/aggregate/` layout.

All input and output locations are defined once in `rat_kp_core/paths.py` (plus the per-pipeline `paths.py` modules). Read those constants rather than hard-coding directories.

`scripts/build_figures.py` reads public aggregate outputs from `results/aggregate/` and writes submission-ready `Fig1`-`Fig4` PNG and TIFF files to `submission_figures/`. Run `scripts/audit_submission_figures.py` after generation to verify dimensions, resolution, color mode, and file integrity. Chemical-space panels additionally require request-only structure-level inputs. The value-free `prepare_matched_panels.py` and `evaluate_manuscript_benchmarks.py` require authorised local PT/RR reconstructions and never distribute those record-level inputs or outputs.

`scripts/build_kp_data_source_bibliography.py` regenerates the aggregate source-provenance inventory distributed through Zenodo when authorised local copies of Kp_Data and its row-aligned source workbook are supplied. It emits bibliographic metadata and aggregate counts only; it does not write record-level source links or experimental values.

## License

Original software is released under the Apache License 2.0. No data or rights in third-party publications are licensed by this repository.
