# RatKpContext

RatKpContext is the software repository supporting the manuscript **Dissecting Tissue-Context Effects in Graph Neural Networks for Rat Kp Prediction: Additive Shifts, Fusion Position, and Extrapolation**.

## Repository boundary

This repository contains code only: model definitions, training and evaluation pipelines, frozen configurations, synthetic tests, statistical analysis, DVI propagation, matched-panel construction, figure generation, and environment specifications. It does not contain the model-development dataset, record-level source links, PT/RR record-level values, real record-level predictions, frozen split assignments, or aggregate study outputs.

Publicly distributable aggregate outputs, run metadata, index-only split assignments, a Kp_Data source bibliography with aggregate contribution counts, and their hashes are provided in the [**RatKpContext Benchmark (RKpC Benchmark)**](https://doi.org/10.5281/zenodo.22671239) Zenodo record. The harmonized model-development dataset and record-level source linkage remain available from Kyeong-Ryoon Lee on reasonable request. Record-level PT/RR source transcriptions are not redistributed.

## Recorded environment

- Python 3.11.15
- PyTorch 2.11.0+cu128
- PyTorch Geometric 2.7.0
- Lightning 2.6.1
- Chemprop 2.2.3
- RDKit 2026.03.1
- Matplotlib 3.10.9
- NVIDIA GeForce RTX 4090, CUDA 12.8

Create the recorded environment with `environment.yml`. The exact top-level package pins are also listed in `requirements.txt`.

## Synthetic verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The public tests use synthetic molecular inputs and exercise all four GNN architectures, five context/fusion conditions, the tissue-intercept control, DVI aggregation, statistical helpers, and value-free matched-panel construction.

## Data-dependent reproduction

1. Obtain the request-only processed model-development data from the corresponding author.
2. Download the RKpC Benchmark from Zenodo and place its `results/` and `splits/` directories at the repository root.
3. Follow `docs/data_schema.md` and `docs/reproducibility_spec.md`.

`scripts/build_figures.py` reads public aggregate outputs from `results/aggregate/` and writes submission-ready `Fig1`-`Fig4` PNG and TIFF files to `submission_figures/`. Run `scripts/audit_submission_figures.py` after generation to verify dimensions, resolution, color mode, and file integrity. Chemical-space panels additionally require request-only structure-level inputs. The value-free `prepare_matched_panels.py` and `evaluate_manuscript_benchmarks.py` require authorised local PT/RR reconstructions and never distribute those record-level inputs or outputs.

`scripts/build_kp_data_source_bibliography.py` regenerates the aggregate source-provenance inventory distributed through Zenodo when authorised local copies of Kp_Data and its row-aligned source workbook are supplied. It emits bibliographic metadata and aggregate counts only; it does not write record-level source links or experimental values.

## License

Original software is released under the Apache License 2.0. No data or rights in third-party publications are licensed by this repository.
