# kernel-fire-precursors (kfp)

Kernel methods for detecting and exploiting distribution-level **meteorological precursors of wildfire occurrence**, plus a leakage-controlled **quantum-vs-classical kernel benchmark**.

Author: **M.S. Shyam Sunder** (ORCID [0000-0002-9964-6199](https://orcid.org/0000-0002-9964-6199)), Dept. of Earth and Atmospheric Sciences, NIT Rourkela.

## Aims
1. **Precursor detection**: test whether the distributions of soil moisture, VPD, temperature, wind and vegetation shift before fire. Uses the MMD two-sample test (permutation) and kernel changepoint detection.
2. **Distribution regression**: use kernel mean embeddings of sub-grid pixel distributions as predictors, fit with kernel ridge regression.
3. **Hyperparameter search**: use a GP-bandit (GP-UCB) to choose kernel bandwidth and lag/window.
4. **Benchmark**: compare simulated quantum kernels against tuned classical kernels and gradient boosting. Uses spatiotemporal blocked CV and reports AUC, PR-AUC, Brier, CSI and an equivalence test.

Regions: Uttarakhand and Odisha (India), and California (USA). No results are claimed yet.

## Structure
```
src/kfp/
  data.py         region bboxes + loader stubs (ERA5-Land, FIRMS, SeasFire)
  features.py     precursor windows (lagged pre-event samples vs baselines)
  mmd.py          unbiased MMD^2 (RBF), median heuristic, permutation test   [working]
  embeddings.py   kernel mean embeddings + distribution regression (KRR)    [working]
  cv.py           spatiotemporal blocked CV splitter with buffer             [working]
  metrics.py      AUC, PR-AUC, Brier, CSI                                    [working]
  changepoint.py  kernel changepoint (sliding-window MMD scan)               [basic]
  bandit.py       GP-UCB search over a discrete grid                         [basic]
  qkernels.py     optional qiskit-machine-learning fidelity kernel (guarded)  [optional]
tests/            pytest on synthetic data
scripts/demo_synthetic.py
```

## Quickstart
```bash
conda env create -f environment.yml && conda activate kfp   # or: pip install -e .[dev]
pytest -q
python scripts/demo_synthetic.py
```

## Data sources
- ERA5-Land (Copernicus CDS): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- NASA FIRMS active fires: https://firms.modaps.eosdis.nasa.gov/
- SeasFire Cube (Zenodo): https://zenodo.org/records/8055879

## Citation
Shyam Sunder, M.S., Tyagi, B., Mahto, S.S. (2026). *[Title TBD]*. arXiv:[XXXX.XXXXX]. (placeholder)

## License
MIT
