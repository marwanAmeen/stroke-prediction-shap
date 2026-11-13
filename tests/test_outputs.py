import json
from pathlib import Path

import pytest

FIG = Path("figures")
EXPECTED_FIGURES = [
    "eda_histograms.png",
    "eda_class_balance.png",
    "eda_stroke_rates.png",
    "rf_force_patientA.png",
    "rf_waterfall_patientA.png",
    "rf_summary.png",
    "rf_bar.png",
    "mlp_summary.png",
    "mlp_bar.png",
    "rf_vs_mlp_importance.png",
    "rf_bmi_artefact.png",
    "patient_top_factors.png",
    "mlp_force_patientA.png",
    "mlp_waterfall_patientA.png",
]


@pytest.mark.parametrize("name", EXPECTED_FIGURES)
def test_figure_exists(name):
    path = FIG / name
    assert path.exists(), path
    assert path.stat().st_size > 5_000


def test_metrics_json():
    m = json.loads((FIG / "metrics.json").read_text())
    for model in ("rf", "mlp"):
        assert set(m[model]) == {"accuracy", "f1_macro", "recall_stroke"}
    assert 0 <= m["patient_a"]["risk"] <= 1
    assert 0 <= m["rf_base_value"] <= 1
    assert len(m["patient_a"]["top_factors"]) == 4


def test_report_evidence_json():
    ev = json.loads((FIG / "report_evidence.json").read_text())
    for key in ("dataset", "split", "models", "shap_verification", "global", "patients", "stability",
                "environment", "starter_script_run"):
        assert key in ev, key
    assert len(ev["dataset"]["sha256"]) == 64
    assert ev["dataset"]["n_rows"] == 5110
    for model in ("rf", "mlp"):
        for k in ("accuracy", "balanced_accuracy", "precision_stroke", "recall_stroke", "f1_stroke", "roc_auc",
                  "average_precision", "confusion"):
            assert k in ev["models"][model], (model, k)
    assert ev["shap_verification"]["rf_treeshap"]["max_abs_residual_all_test"] < 1e-6
    assert ev["shap_verification"]["mlp_kernelshap"]["max_abs_residual"] < 1e-3
    assert ev["starter_script_run"]["return_code"] == 0
    assert len(ev["patients"]["A"]["rf"]["top3"]) == 3
