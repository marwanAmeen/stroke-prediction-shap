# Stroke prediction with SHAP

Explaining stroke classifiers on the Kaggle *Stroke Prediction Dataset* (fedesoriano):

- Random forest (`min_samples_leaf=20`) explained with **TreeSHAP** (local force/waterfall plot, global summary plot)
- MLP explained with **Kernel SHAP**

## Layout

| File | Purpose |
|---|---|
| `stroke_xai.py` | Whole analysis as a jupytext percent-format notebook (EDA, both models, SHAP, evidence export) |
| `xai_helpers.py` | Median-BMI preprocessing, patient selection, metrics, plain-language labels |
| `tests/` | Helper tests and output checks |
| `xai-series-master/utils.py`, `00_data_exploration.py` | Starter code from the course `xai-series` repo, unmodified, used for the data loader and the EDA run |

The starter files are third-party course material. They are included only so the pipeline runs; they are not covered by this repository's own work.

## Data (not included)

Download `healthcare-dataset-stroke-data.csv` from
<https://www.kaggle.com/fedesoriano/stroke-prediction-dataset> and place it in both:

- `dataset/healthcare-dataset-stroke-data.csv`
- `xai-series-master/data/healthcare-dataset-stroke-data.csv`

## Run

Python 3.12 is recommended (`shap`/`numba` do not support newer versions yet).

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python -m ipykernel install --sys-prefix --name stroke-xai
.venv\Scripts\jupytext --from py:percent --to ipynb stroke_xai.py
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=stroke-xai stroke_xai.ipynb
.venv\Scripts\python -m pytest -q
```

The notebook takes about 3 minutes and writes plots and `metrics.json` / `report_evidence.json` to `figures/`.
`tests/test_outputs.py` checks those generated files, so run it after the notebook.
