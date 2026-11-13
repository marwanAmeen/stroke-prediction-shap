# %% [markdown]
# # Explaining a stroke-risk model with SHAP
# Tasks 1–4 and 6 of the XAI assignment.

# %%
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "xai-series-master")
from utils import DataLoader  # noqa: E402
from xai_helpers import classification_metrics, mlp_preprocess, plain_label, select_patients  # noqa: E402

RANDOM_STATE = 2021
DATA_PATH = "dataset/healthcare-dataset-stroke-data.csv"
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def save(name):
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.show()


print("Python", platform.python_version(), "| shap", shap.__version__, "| sklearn", sklearn.__version__)
shap.initjs()

# %% [markdown]
# ## Task 1: Exploratory data analysis
# This reproduces `00_data_exploration.py` and adds class balance and stroke-rate breakdowns.

# %%
raw = pd.read_csv(DATA_PATH)
print(raw.shape)
raw.head()

# %%
raw.info()

# %%
raw.describe()

# %%
raw.isna().sum()

# %%
cols = [c for c in raw.columns if c != "id"]
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, cols):
    if pd.api.types.is_numeric_dtype(raw[col]):
        raw[col].hist(ax=ax, bins=30)
    else:
        raw[col].value_counts().plot.bar(ax=ax)
    ax.set_title(col)
for ax in axes.flat[len(cols):]:
    ax.axis("off")
fig.tight_layout()
save("eda_histograms.png")

# %%
counts = raw["stroke"].value_counts().sort_index()
ax = counts.plot.bar(color=["tab:blue", "tab:red"])
ax.set_xticklabels(["No stroke", "Stroke"], rotation=0)
ax.set_title(f"Class balance: {raw['stroke'].mean():.1%} of patients had a stroke")
save("eda_class_balance.png")

# %%
groups = {
    "Age band": pd.cut(raw["age"], [0, 20, 40, 60, 80, 100]),
    "Hypertension": raw["hypertension"],
    "Heart disease": raw["heart_disease"],
    "Glucose band (mg/dL)": pd.cut(raw["avg_glucose_level"], [0, 100, 140, 200, 300]),
}
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, g) in zip(axes, groups.items()):
    (raw.groupby(g, observed=True)["stroke"].mean() * 100).plot.bar(ax=ax, color="tab:red")
    ax.set_title(name)
    ax.set_ylabel("Stroke rate (%)")
    ax.set_xlabel("")
fig.tight_layout()
save("eda_stroke_rates.png")

# %%
data_loader = DataLoader()
data_loader.load_dataset(DATA_PATH)
data_loader.preprocess_data()
data_loader.data.head()

# %% [markdown]
# **Findings:**
# - 5,110 patients and 12 columns; `bmi` has 201 missing values (3.9%). The starter code fills them with 0;
#   we use the training median instead (see the start of Tasks 2–4 for why).
# - Strong class imbalance: 249 patients (4.9%) had a stroke, so we oversample the training set and report F1 and stroke recall, not just accuracy.
# - Stroke rate rises steeply with age: under 1% below 40, 4.1% at 40–60, 13.0% at 60–80 and 19.8% over 80.
#   It is also higher with hypertension (13.3% vs 4.0%), heart disease (17.0% vs 4.2%) and glucose above
#   140 mg/dL (9.6–12.9% vs about 3.7% below 140).
# - 30% of `smoking_status` values are "Unknown", which limits how much we can read into smoking.

# %% [markdown]
# ## Tasks 2–4: RandomForest explained with TreeSHAP
#
# ### Why we don't use the starter's BMI = 0
# The starter `utils.DataLoader` fills missing BMI with 0. We first train the RF exactly as the starter
# code would and look at the SHAP values for BMI.

# %%
Xs_train, Xs_test, ys_train, ys_test = data_loader.get_data_split()
np.random.seed(RANDOM_STATE)
Xs_train, ys_train = data_loader.oversample(Xs_train, ys_train)
Xs_train, Xs_test, ys_train = Xs_train.astype(float), Xs_test.astype(float), ys_train.astype(int)
rf_starter = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1).fit(Xs_train, ys_train)
sv_starter = shap.TreeExplainer(rf_starter)(Xs_test)[..., 1]

shap.plots.scatter(sv_starter[:, "bmi"], show=False)
plt.title("Starter preprocessing: SHAP value of BMI (missing BMI set to 0)")
save("rf_bmi_artefact.png")

missing = raw["bmi"].isna()
bmi_sv_starter = sv_starter[:, "bmi"].values
recorded_test = Xs_test["bmi"].to_numpy() > 0
print("Starter-preprocessed RF:", {k: round(v, 3) for k, v in classification_metrics(ys_test, rf_starter.predict(Xs_test)).items()})
print(f"Stroke rate, BMI missing:  {raw.loc[missing, 'stroke'].mean():.1%} ({missing.sum()} patients)")
print(f"Stroke rate, BMI recorded: {raw.loc[~missing, 'stroke'].mean():.1%} ({(~missing).sum()} patients)")
print(f"Test patients with a recorded BMI whose BMI SHAP is negative: {np.mean(bmi_sv_starter[recorded_test] < 0):.0%} "
      f"(mean {bmi_sv_starter[recorded_test].mean():+.3f}); BMI = 0 patients: mean {bmi_sv_starter[~recorded_test].mean():+.3f}")

# %% [markdown]
# **The leak.** Missing BMI is not random in this dataset: 19.9% of the 201 patients with a missing BMI had a
# stroke, against 4.3% of the 4,909 with a recorded BMI. With missing BMI set to 0, the RF learns
# "BMI = 0 means stroke". In the plot, the column of dots at BMI = 0 reaches SHAP values of about +0.16
# (mean +0.048), while 100% of test patients with a recorded BMI get a negative BMI SHAP value (mean −0.089).
# An explanation built on this tells a doctor that BMI matters because it was *not recorded*. That is about
# how the data was collected, not about the patient's body.
#
# **Decision (a deliberate deviation from the starter code).** For Tasks 2–4 and Task 6 we fill missing BMI with
# the training-set median instead of 0. The columns, split and oversampling stay as in the starter code.

# %% [markdown]
# ### The data used in Tasks 2–4
# One-hot encoding, missing BMI filled with the **training-set median**, no scaling, and random oversampling
# of the training set only (`xai_helpers.mlp_preprocess`, then the starter's `oversample`). The columns, the
# train/test split and the oversampling are the same as in the starter code; only the BMI imputation differs.
# The Task 6 MLP uses exactly the same data. The forest itself uses `min_samples_leaf=20` (explained below).

# %%
X_train, X_test, y_train, y_test = mlp_preprocess(raw, random_state=RANDOM_STATE)
assert X_test.index.equals(Xs_test.index) and list(X_test.columns) == list(Xs_test.columns)
bmi_recorded = raw.loc[X_test.index, "bmi"].notna().to_numpy()
np.random.seed(RANDOM_STATE)
X_train, y_train = DataLoader().oversample(X_train, y_train)
X_train, y_train = X_train.astype(float), y_train.astype(int)
print(X_train.shape, X_test.shape, y_train.value_counts().to_dict())

# %% [markdown]
# ### Why `min_samples_leaf=20`
# First, a RandomForest with default settings on this median-BMI data. TreeSHAP is run on the first 200 test
# patients only, which is enough to show the pattern.

# %%
rf_default = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
def_pred = rf_default.predict(X_test)
print("Default RF:", {k: round(v, 3) for k, v in classification_metrics(y_test, def_pred).items()},
      f"| finds {int(((def_pred == 1) & (y_test == 1)).sum())} of {int(y_test.sum())} strokes")
print(f"Mean default-RF score: training stroke rows (incl. duplicates) "
      f"{rf_default.predict_proba(X_train)[:, 1][y_train.to_numpy() == 1].mean():.3f}; "
      f"test patients {rf_default.predict_proba(X_test)[:, 1].mean():.3f}")
sub = slice(0, 200)
sv_def = shap.TreeExplainer(rf_default)(X_test.iloc[sub])[..., 1]
print(f"Default RF, first 200 test patients: base value {sv_def.base_values[0]:.3f}; "
      f"{np.mean(sv_def[:, 'bmi'].values[bmi_recorded[sub]] < 0):.1%} of recorded-BMI patients get a negative BMI SHAP, "
      f"{np.mean(sv_def[:, 'avg_glucose_level'].values < 0):.0%} a negative glucose SHAP")

# %% [markdown]
# **Memorisation.** By default each tree keeps splitting until its leaves are pure. Random oversampling makes
# many exact copies of every training stroke patient, so the trees can fence those copies off in their own
# leaves. The default forest scores the training stroke rows 1.000 on average, but the test patients only
# 0.068, and it finds only 3 of the 50 test strokes (recall 0.06). The SHAP base value is the training average
# (0.501), so almost every test patient has to be explained as "far below average", and the continuous features
# absorb that gap: on the first 200 test patients, 99.5% of those with a recorded BMI get a negative BMI SHAP
# value and 98% get a negative glucose SHAP value. BMI and glucose then look protective for nearly everyone.
#
# **Decision.** The main RF uses `min_samples_leaf=20`: every leaf must hold at least 20 training rows, which
# makes it much harder for a leaf to contain only the copies of one patient. This is a single regularisation
# choice, picked as a simple fix. It was not tuned (no search over values and no validation set), and other
# values might work as well or better.

# %%
rf = RandomForestClassifier(min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)
rf_metrics = classification_metrics(y_test, rf_pred)
print(f"Test: {len(y_test)} patients, {int(y_test.sum())} strokes; RF finds {int(((rf_pred == 1) & (y_test == 1)).sum())} "
      f"with {int(((rf_pred == 1) & (y_test == 0)).sum())} false alarms")
rf_metrics

# %%
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, display_labels=["No stroke", "Stroke"])
plt.show()

# %% [markdown]
# With `min_samples_leaf=20` the RF finds 29 of the 50 test strokes (recall 0.58), with 181 false alarms.
# Accuracy (0.802) is lower than the default forest's (0.941), but that accuracy was misleading: 972 of the
# 1,022 test patients (95%) had no stroke, so predicting "no stroke" almost always scores well. Macro F1 is 0.555.
#
# Because the training set was oversampled to 50/50, the model's average prediction (the SHAP base value
# below, 0.501) is about 0.50, not the real 4.9% stroke rate. Its outputs rank patients by risk but are not
# calibrated probabilities.
#
# ### Task 2: SHAP explainer
# `TreeExplainer` computes exact Shapley values for tree ensembles. We explain the probability of
# class 1 (stroke).

# %%
explainer = shap.TreeExplainer(rf)
sv = explainer(X_test)
sv1 = sv[..., 1]
print("SHAP values shape (patients, features, classes):", sv.values.shape)

recon = sv1.base_values + sv1.values.sum(axis=1)
assert np.allclose(recon, rf_proba, atol=1e-6)
print(f"Additivity holds: base value {sv1.base_values[0]:.3f} + sum of SHAP values = predicted probability")
print(f"Mean RF score: test patients {rf_proba.mean():.3f}; training stroke rows (incl. duplicates) "
      f"{rf.predict_proba(X_train)[:, 1][y_train.to_numpy() == 1].mean():.3f}")

# %% [markdown]
# Patients A and B are chosen only among test patients whose BMI was actually recorded, so the case study
# is not explained by an imputed value.

# %%
eligible = np.flatnonzero(bmi_recorded)
sel = select_patients(y_test.to_numpy()[eligible], rf_proba[eligible])
patients = {"A": int(eligible[sel["A"]]), "B": int(eligible[sel["B"]]), "a_is_tp": sel["a_is_tp"]}
A, B = patients["A"], patients["B"]
print(patients, "| P(stroke) A =", round(rf_proba[A], 3), "B =", round(rf_proba[B], 3))
raw.loc[X_test.index[[A, B]]]

# %% [markdown]
# ### Task 3: Local explanation (force plots)
# Red features push this patient's predicted risk above the base value (the average prediction);
# blue features push it down.

# %%
shap.plots.force(sv1[A])

# %%
shap.plots.force(sv1[B])

# %%
shap.plots.force(sv1[A].base_values, sv1[A].values, X_test.iloc[A].round(2), matplotlib=True, show=False,
                 figsize=(24, 3), contribution_threshold=0.1)
save("rf_force_patientA.png")

# %%
shap.plots.waterfall(sv1[A], show=False)
save("rf_waterfall_patientA.png")

# %%
shap.plots.waterfall(sv1[B], show=False)
plt.show()

# %%
for name, i_p in (("A", A), ("B", B)):
    s_p = pd.Series(sv1[i_p].values, index=X_test.columns)
    print(f"Patient {name}: P(stroke) = {rf_proba[i_p]:.2f}, stroke = {y_test.iloc[i_p]}")
    print(s_p.reindex(s_p.abs().sort_values(ascending=False).index).head(6).round(3).to_string())
    print(f"  age {s_p['age']:+.3f}, glucose {s_p['avg_glucose_level']:+.3f}, hypertension {s_p['hypertension']:+.3f}")
    print()

# %% [markdown]
# **Findings:**
# - **Patient A** (test position 357, a 67-year-old man with heart disease, no hypertension, glucose
#   228.7 mg/dL, a recorded BMI of 36.6, former smoker) is a true positive with a score of 0.81, against a base
#   value of 0.501. High glucose (+0.098), age (+0.089) and heart disease (+0.081) are the largest pushes up,
#   and being a former smoker adds +0.021. This is the picture a clinician would expect.
# - His BMI of 36.6 still lowers his score slightly (−0.026). Obesity is a known stroke risk factor, so this
#   small effect is not clinically plausible; Task 4 shows it is part of a wider BMI pattern in this model.
# - The forest no longer memorises the training strokes: it scores the training stroke rows 0.777 on average
#   (the default forest: 1.000) and the test patients 0.258.
# - **Patient B** (test position 113, a 12-year-old boy, rural, glucose 117.0 mg/dL, BMI 18.1, smoking status
#   "Unknown") is a true negative with a score of 0.00. Age dominates (−0.233). Next come work type "children"
#   (−0.046), never married (−0.041 and −0.039 for the two marital-status columns), BMI (−0.039) and glucose
#   (−0.034). Young age lowering the risk is plausible. Our reading is that work type and marital status mostly
#   stand in for "is a child" here, so they repeat the age effect rather than add new information.

# %% [markdown]
# ### Task 4: Global explanation (summary plots)
# Each dot is one test patient. The x-position is that feature's SHAP value; the colour is the
# feature's value (red = high).

# %%
shap.plots.beeswarm(sv1, max_display=12, show=False)
save("rf_summary.png")

# %%
shap.plots.bar(sv1, max_display=12, show=False)
save("rf_bar.png")

# %%
rf_rank = pd.Series(np.abs(sv1.values).mean(0), index=X_test.columns).sort_values(ascending=False)
print(rf_rank.head(8).round(3).to_string())
bmi_sv = sv1[:, "bmi"].values
bmi_val = X_test["bmi"].to_numpy()
for name, mask in (("BMI < 25", bmi_val < 25), ("BMI 25-30", (bmi_val >= 25) & (bmi_val < 30)), ("BMI >= 30", bmi_val >= 30)):
    print(f"{name}: mean BMI SHAP {bmi_sv[mask & bmi_recorded].mean():+.3f} ({(mask & bmi_recorded).sum()} recorded-BMI patients)")
age_sv = sv1[:, "age"].values
print(f"Recorded-BMI patients with negative BMI SHAP: {np.mean(bmi_sv[bmi_recorded] < 0):.1%}; "
      f"median-imputed patients ({(~bmi_recorded).sum()}): mean BMI SHAP {bmi_sv[~bmi_recorded].mean():+.3f}")
print(f"Glucose SHAP negative for {np.mean(sv1[:, 'avg_glucose_level'].values < 0):.0%} of test patients")
print(f"Age SHAP range {age_sv.min():+.2f} to {age_sv.max():+.2f}")
for c in ["avg_glucose_level", "hypertension", "heart_disease"]:
    v, s = X_test[c].to_numpy(), sv1[:, c].values
    hi = v > 140 if c == "avg_glucose_level" else v == 1
    print(f"{c}: mean SHAP {s[hi].mean():+.3f} when high/present vs {s[~hi].mean():+.3f} otherwise")

# %% [markdown]
# **Findings:**
# - Age is the dominant global driver (mean |SHAP| 0.194). Older patients get positive SHAP values of up to
#   +0.30, and young patients get negative values down to −0.34.
# - Glucose (0.059) and BMI (0.030) come next, then hypertension (0.026), being a former smoker (0.021) and
#   heart disease (0.017).
# - The directions now match clinical knowledge. Glucose above 140 mg/dL pushes the risk up (mean SHAP +0.052,
#   against −0.057 at or below 140), and so do hypertension (+0.072 vs −0.021) and heart disease (+0.072 vs −0.014).
#   87% of test patients still get a negative glucose SHAP. Our reading is that most patients have normal
#   glucose and the base value (0.501) is inflated by oversampling, so "normal" reads as below average.
# - BMI is weaker and still not clean. 81.8% of patients with a recorded BMI get a negative BMI SHAP value. The
#   effect is not monotone: mean −0.030 below 25, +0.000 at 25–30 and −0.041 at 30 or above, so in this model a
#   high BMI does not raise the risk.
# - The 42 test patients with a median-imputed BMI get a mean BMI SHAP of +0.033. *Hypothesis (not tested
#   here):* the forest may still partly recognise the exact imputed value as "BMI was missing", a much weaker
#   version of the BMI = 0 leak (which gave +0.048 under the starter preprocessing).

# %% [markdown]
# ## Task 6: MLP explained with Kernel SHAP
# The MLP is trained on exactly the same oversampled training set as the RF (median-imputed BMI) and
# evaluated on the same test patients in the same positions, so patients A and B line up. A neural network
# needs scaled inputs, so a `StandardScaler` sits inside the pipeline; SHAP still sees raw feature values.

# %%
assert X_test.index.equals(Xs_test.index) and len(X_test) == len(rf_proba)

mlp = Pipeline([
    ("scale", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=500,
                          random_state=RANDOM_STATE)),
])
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_metrics = classification_metrics(y_test, mlp_pred)
print(f"MLP finds {int(((mlp_pred == 1) & (y_test == 1)).sum())} of {int(y_test.sum())} strokes "
      f"with {int(((mlp_pred == 1) & (y_test == 0)).sum())} false alarms")
mlp_metrics

# %%
ConfusionMatrixDisplay.from_predictions(y_test, mlp_pred, display_labels=["No stroke", "Stroke"])
plt.show()

# %% [markdown]
# `KernelExplainer` is model-agnostic: it only calls `predict_proba`. It estimates Shapley values by
# sampling feature coalitions and replacing "missing" features with a background set (20 k-means
# centroids of the training data). Because it is slow, we explain the first 100 test patients plus
# patients A and B.

# %%
cols = list(X_train.columns)


def f(X):
    return mlp.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]


kexp = shap.KernelExplainer(f, shap.kmeans(X_train, 20))
explain_pos = sorted(set(range(100)) | {A, B})
X_explain = X_test.iloc[explain_pos]
_t0 = time.perf_counter()
kvals = np.asarray(kexp.shap_values(X_explain, nsamples=500, silent=True)).reshape(len(X_explain), len(cols))
t_kernel = time.perf_counter() - _t0
k_base = float(np.ravel(kexp.expected_value)[0])
ksv = shap.Explanation(values=kvals, base_values=np.full(len(X_explain), k_base),
                       data=X_explain.to_numpy(), feature_names=cols)
pos = {p: i for i, p in enumerate(explain_pos)}

assert np.allclose(ksv.base_values + ksv.values.sum(axis=1), f(X_explain), atol=1e-3)
print("Kernel SHAP additivity holds (approximately)")

# %%
shap.plots.force(ksv[pos[A]])

# %%
shap.plots.waterfall(ksv[pos[A]], show=False)
save("mlp_waterfall_patientA.png")
k_a = pd.Series(ksv.values[pos[A]], index=cols)
print(f"MLP patient A: P(stroke) = {f(X_test.iloc[[A]])[0]:.2f}, base value {k_base:.2f}")
print(k_a.reindex(k_a.abs().sort_values(ascending=False).index).head(6).round(3).to_string())

# %%
shap.plots.force(ksv[pos[A]].base_values, ksv[pos[A]].values, X_test.iloc[A].round(2), matplotlib=True, show=False,
                 figsize=(24, 3), contribution_threshold=0.1)
save("mlp_force_patientA.png")

# %%
shap.plots.beeswarm(ksv, max_display=12, show=False)
save("mlp_summary.png")

# %%
shap.plots.bar(ksv, max_display=12, show=False)
save("mlp_bar.png")

# %% [markdown]
# ### RF vs MLP: do the two models rely on the same features?
# Importance is the share of total mean |SHAP| on the same patients, so the two models are comparable.

# %%
rf_imp = pd.Series(np.abs(sv1.values[explain_pos]).mean(0), index=X_test.columns)
mlp_imp = pd.Series(np.abs(ksv.values).mean(0), index=cols)
imp = pd.DataFrame({
    "RandomForest (TreeSHAP)": rf_imp / rf_imp.sum(),
    "MLP (Kernel SHAP)": mlp_imp / mlp_imp.sum(),
})
top = imp.max(axis=1).sort_values(ascending=False).index[:12]
imp.loc[top][::-1].plot.barh(figsize=(8, 7))
plt.xlabel("Share of total mean |SHAP|")
plt.title("Feature importance: RandomForest vs MLP")
save("rf_vs_mlp_importance.png")
print(len(explain_pos), "explained patients")
print((imp.loc[["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]] * 100).round(0).to_string())
imp.loc[top].rank(ascending=False).astype(int)

# %% [markdown]
# **TreeSHAP vs Kernel SHAP:**
# - TreeSHAP is exact and fast, but it only works for tree models. Kernel SHAP works on any model but
#   is a sampled approximation: it is slow, its results depend on the background set, and it treats
#   features as independent.
# - Additivity holds exactly for TreeSHAP (to 1e-6) and only approximately for Kernel SHAP (to 1e-3).
#
# **Performance on the same 1,022 test patients (50 strokes), same training data:**
#
# | Model | Accuracy | Macro F1 | Stroke recall |
# |---|---|---|---|
# | RandomForest (starter BMI = 0, default settings) | 0.944 | 0.547 | 0.08 (4 of 50) |
# | RandomForest (median BMI, default settings) | 0.941 | 0.530 | 0.06 (3 of 50) |
# | RandomForest (median BMI, `min_samples_leaf=20`), used in Tasks 2–4 | 0.802 | 0.555 | 0.58 (29 of 50) |
# | MLP (median BMI) | 0.795 | 0.528 | 0.44 (22 of 50) |
#
# The final RF beats the MLP on all three measures, with about the same number of false alarms (181 vs 182).
# The first row uses the starter data (BMI = 0); the other three share the median-BMI training set.
#
# **Where the explanations agree and disagree (102 explained patients):**
# - Both models rank age first, with RF 46% and MLP 32% of total importance.
# - The RF puts most of the rest on glucose (2nd, 14%), BMI (3rd, 7%) and hypertension (4th, 6%), with being a
#   former smoker 5th. Age, glucose and BMI together carry about 67% of its importance.
# - The MLP spreads importance over the one-hot columns. "Never smoked" is 2nd, BMI 3rd (8%) and male gender 4th,
#   while glucose is only 8th (5%), hypertension (1%) is last of the 12 shown, and heart disease gets about 0%.
#   Its "never smoked" SHAP values are positive, which is unlikely to be causal. *Hypothesis (not tested here):*
#   the column mainly separates adults from the "Unknown" smoking group, which includes many children.
# - For patient A the MLP gives a score of 0.98 (base value 0.30). Being male (+0.189, partly offset by
#   `gender_Female` −0.058), age (+0.175), being a former smoker (+0.150), BMI (+0.087) and glucose (+0.071)
#   drive it. Both models now push patient A up through age and glucose. They disagree on BMI (RF −0.026,
#   MLP +0.087) and on heart disease (RF +0.081; not in the MLP's top six). The MLP's large weight on sex is
#   questionable.
# - Some of the differences lower down may also be Kernel SHAP sampling noise (102 patients, 500 samples each).

# %% [markdown]
# ## Plain-language summary of patient A

# %%
a = sv1[A]
by_label = pd.Series(a.values, index=[plain_label(c) for c in X_test.columns]).groupby(level=0).sum()
top4 = by_label.reindex(by_label.abs().sort_values(ascending=False).index[:4])
effects = top4 * 100
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(effects.index[::-1], effects.values[::-1],
        color=["#d62728" if e > 0 else "#1f77b4" for e in effects.values[::-1]])
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Change in estimated risk (percentage points)")
ax.set_title("What raises and lowers your estimated risk" if (effects < 0).any() else "What raises your estimated risk")
fig.tight_layout()
save("patient_top_factors.png")

# %%
summary = {
    "rf": rf_metrics,
    "mlp": mlp_metrics,
    "rf_default": classification_metrics(y_test, def_pred),
    "rf_min_samples_leaf": rf.min_samples_leaf,
    "rf_base_value": float(sv1.base_values[0]),
    "n_test": int(len(y_test)),
    "stroke_rate": float(raw["stroke"].mean()),
    "patient_a": {
        "risk": float(rf_proba[A]),
        "is_true_positive": patients["a_is_tp"],
        "age": float(X_test.iloc[A]["age"]),
        "bmi": float(X_test.iloc[A]["bmi"]),
        "bmi_effect_pp": round(float(sv1[A, "bmi"].values * 100), 1),
        "top_factors": [{"label": k, "effect_pp": round(float(v * 100), 1)} for k, v in top4.items()],
    },
}
(FIG / "metrics.json").write_text(json.dumps(summary, indent=2))
summary


# %% [markdown]
# ## Report evidence
# Everything in this section only *measures* the models and explainers built above and writes the numbers the
# written report cites to `figures/report_evidence.json`. It trains nothing new for the main results (the
# only extra models are the stability checks at the end). Nothing here is tuned on the test set.

# %%
import datetime
import filecmp
import hashlib
import os
import subprocess

import imblearn
import matplotlib
import scipy
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from xai_helpers import CATEGORICAL_COLS

ev = {}
feat = list(X_train.columns)
y_te = y_test.to_numpy()


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --- Starter script, run exactly as the brief says (from xai-series-master, headless backend) ---
_started = datetime.datetime.now().isoformat(timespec="seconds")
_proc = subprocess.run([sys.executable, "00_data_exploration.py"], cwd="xai-series-master",
                       env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True)
_warn = sorted({ln.strip().split("\\")[-1] for ln in _proc.stderr.splitlines() if "Warning" in ln})
ev["starter_script_run"] = {
    "command": "cd xai-series-master; MPLBACKEND=Agg python 00_data_exploration.py",
    "date": _started, "return_code": _proc.returncode, "status": "success" if _proc.returncode == 0 else "failed",
    "stdout_first_line": _proc.stdout.splitlines()[0] if _proc.stdout else "",
    "stderr_distinct_warning_lines": _warn,
    "stderr_line_count": len(_proc.stderr.splitlines()),
    "script_modified": False,
}

# --- Dataset checks ---
csv_path = Path(DATA_PATH)
starter_csv = Path("xai-series-master/data/healthcare-dataset-stroke-data.csv")


def describe(s):
    s = s.dropna()
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std()), "median": float(med),
            "q1": float(q1), "q3": float(q3), "min": float(s.min()), "max": float(s.max())}


NUM = ["age", "avg_glucose_level", "bmi"]
ev["dataset"] = {
    "path": DATA_PATH,
    "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    "bytes": csv_path.stat().st_size,
    "identical_to_starter_copy": bool(filecmp.cmp(csv_path, starter_csv, shallow=False)),
    "file_modified_local": datetime.datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat(timespec="seconds"),
    "starter_copy_modified_local": datetime.datetime.fromtimestamp(starter_csv.stat().st_mtime).isoformat(timespec="seconds"),
    "n_rows": int(raw.shape[0]), "n_cols": int(raw.shape[1]), "columns": list(raw.columns),
    "duplicate_rows": int(raw.duplicated().sum()), "duplicate_ids": int(raw["id"].duplicated().sum()),
    "bmi_missing_token_in_file": "N/A (read as NaN by pandas)",
    "missing": {c: {"n": int(raw[c].isna().sum()), "pct": float(raw[c].isna().mean() * 100)} for c in raw.columns},
    "class_counts": {"n0": int((raw["stroke"] == 0).sum()), "n1": int((raw["stroke"] == 1).sum()),
                     "pct_positive": float(raw["stroke"].mean() * 100)},
    "numeric": {c: describe(raw[c]) for c in NUM},
    "numeric_by_stroke": {str(k): {c: describe(g[c]) for c in NUM} for k, g in raw.groupby("stroke")},
    "plausibility": {"age_below_1": int((raw["age"] < 1).sum()), "age_below_18": int((raw["age"] < 18).sum()),
                     "bmi_above_50": int((raw["bmi"] > 50).sum()),
                     "glucose_above_250": int((raw["avg_glucose_level"] > 250).sum())},
    "binary": {c: {"n1": int(raw[c].sum()), "pct1": float(raw[c].mean() * 100)} for c in ["hypertension", "heart_disease"]},
    "categorical": {c: {str(k): {"n": int(v), "pct": float(v / len(raw) * 100)} for k, v in raw[c].value_counts().items()}
                    for c in CATEGORICAL_COLS},
    "work_type_spelling_in_file": sorted(raw["work_type"].unique().tolist()),
}

# --- Split facts ---
Xtr0, Xte0, ytr0, yte0 = mlp_preprocess(raw, random_state=RANDOM_STATE)
ev["split"] = {
    "method": "sklearn train_test_split(test_size=0.20, random_state=2021) from the starter utils.py; NOT stratified; no validation set",
    "seed": RANDOM_STATE, "stratified": False, "validation_set": False,
    "n_total": int(len(raw)),
    "n_train_before_oversampling": int(len(Xtr0)), "train_pos_before": int(ytr0.sum()),
    "train_pos_pct_before": float(ytr0.mean() * 100),
    "n_train_after_oversampling": int(len(X_train)),
    "train_class_counts_after": {str(k): int(v) for k, v in y_train.value_counts().items()},
    "oversampling": ("imblearn RandomOverSampler(sampling_strategy='minority'), training set only; the starter gives it "
                     "no random_state, so the global NumPy RNG is seeded with 2021 just before it"),
    "n_test": int(len(y_test)), "test_pos": int(y_test.sum()), "test_prevalence_pct": float(y_test.mean() * 100),
    "test_fraction": float(len(y_test) / len(raw)), "train_fraction": float(len(Xtr0) / len(raw)),
    "bmi_median_used_for_imputation": float(raw.loc[Xtr0.index, "bmi"].median()),
    "n_test_bmi_imputed": int((~bmi_recorded).sum()),
}


# --- Held-out performance at threshold 0.5, both models ---
def case_type(y, pred):
    return {(1, 1): "true positive", (0, 1): "false positive", (0, 0): "true negative", (1, 0): "false negative"}[(int(y), int(pred))]


def model_block(proba, pred):
    p, r, f1, sup = precision_recall_fscore_support(y_te, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    edges = np.linspace(0, 1, 6)
    which = np.clip(np.digitize(proba, edges[1:-1]), 0, 4)
    bins = [{"range": [float(edges[i]), float(edges[i + 1])], "n": int((which == i).sum()),
             "mean_score": float(proba[which == i].mean()) if (which == i).any() else None,
             "observed_stroke_fraction": float(y_te[which == i].mean()) if (which == i).any() else None}
            for i in range(5)]
    return {
        "threshold": 0.5,
        "accuracy": float((pred == y_te).mean()), "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "precision_stroke": float(p[0]), "recall_stroke": float(r[0]), "f1_stroke": float(f1[0]), "support_stroke": int(sup[0]),
        "f1_macro": float(classification_metrics(y_te, pred)["f1_macro"]),
        "roc_auc": float(roc_auc_score(y_te, proba)), "average_precision": float(average_precision_score(y_te, proba)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n_predicted_positive": int(pred.sum()), "test_prevalence": float(y_te.mean()),
        "calibration": {"kind": "descriptive only (5 equal-width score bins, Brier score); no recalibration fitted",
                        "brier": float(brier_score_loss(y_te, proba)),
                        "brier_always_predict_prevalence": float(brier_score_loss(y_te, np.full(len(y_te), y_te.mean()))),
                        "mean_score": float(proba.mean()), "observed_prevalence": float(y_te.mean()), "bins": bins},
    }


mlp_proba = mlp.predict_proba(X_test)[:, 1]
ev["models"] = {
    "rf": {"description": "RandomForestClassifier(min_samples_leaf=20, random_state=2021, n_jobs=-1); other settings default (100 trees)",
           "params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in rf.get_params().items()},
           **model_block(rf_proba, rf_pred)},
    "mlp": {"description": "Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=500, random_state=2021))",
            "n_iter": int(mlp.named_steps["mlp"].n_iter_),
            "params": {k: (v if isinstance(v, (int, float, str, type(None), tuple)) else str(v))
                       for k, v in mlp.named_steps["mlp"].get_params().items()},
            **model_block(mlp_proba, mlp_pred)},
}
ev["calibration_note"] = ("Calibration was inspected descriptively only. Training was oversampled to 50/50, so scores are "
                          "not probabilities of stroke at the real 4.9% prevalence; no calibration method was fitted.")

# --- SHAP verification ---
t0 = time.perf_counter()
sv_timed = shap.TreeExplainer(rf)(X_test)
t_tree = time.perf_counter() - t0
sv_t1 = sv_timed[..., 1]
rf_resid_all = np.abs(sv_t1.base_values + sv_t1.values.sum(1) - rf_proba)
rf_resid_sub = rf_resid_all[explain_pos]
k_out = f(X_explain)
k_resid = np.abs(ksv.base_values + ksv.values.sum(1) - k_out)
bg = kexp.data
ev["shap_verification"] = {
    "versions": {"shap": shap.__version__, "sklearn": sklearn.__version__},
    "rf_treeshap": {
        "explainer": "shap.TreeExplainer(rf), tree_path_dependent (no background data), model_output=" + str(explainer.model_output),
        "feature_names_match_X_test_order": bool(list(sv_timed.feature_names) == list(X_test.columns)),
        "n_features": len(feat), "shap_values_shape_all_classes": list(sv_timed.values.shape),
        "classes_": rf.classes_.tolist(), "class_index_explained": 1,
        "explained_output": "predict_proba(X)[:, 1] = P(stroke = 1) as the forest's score, probability scale (0-1)",
        "base_value": float(sv_t1.base_values[0]), "expected_value_both_classes": np.ravel(explainer.expected_value).tolist(),
        "mean_test_score": float(rf_proba.mean()),
        "n_explained": int(len(X_test)), "n_explained_positive": int(y_test.sum()),
        "max_abs_residual_all_test": float(rf_resid_all.max()), "tolerance": 1e-6,
        "max_abs_residual_explained_102_rows": float(rf_resid_sub.max()),
        "runtime_seconds": float(t_tree), "runtime_note": "re-timed run of TreeExplainer(rf)(X_test); results identical to Task 2",
    },
    "mlp_kernelshap": {
        "explainer": "shap.KernelExplainer(f, background); f = mlp.predict_proba(DataFrame(X, columns=feature_names))[:, 1]",
        "feature_names_match_X_test_order": bool(list(X_explain.columns) == feat),
        "n_features": len(feat), "shap_values_shape": list(ksv.values.shape),
        "classes_": mlp.classes_.tolist(), "class_index_explained": 1,
        "explained_output": "P(stroke = 1) from the full pipeline (StandardScaler + MLP), probability scale; SHAP sees unscaled features",
        "base_value": float(k_base),
        "base_value_equals_weighted_background_mean_output":
            bool(np.isclose(k_base, np.average(f(pd.DataFrame(bg.data, columns=feat)), weights=bg.weights))),
        "background": (f"shap.kmeans(X_train, 20): {len(bg.weights)} k-means centroids of the oversampled training set "
                       "(KMeans random_state=0, n_init=10, values rounded to observed training values), weights = cluster sizes"),
        "background_n": int(len(bg.weights)),
        "nsamples": 500, "l1_reg": "default 'num_features(10)' (at most 10 features can be non-zero per explained row)",
        "sampling_seed": ("global NumPy RNG (np.random.seed(2021) was set before oversampling and not reset); "
                          "sampling is deterministic given the notebook's execution order"),
        "n_explained": int(len(X_explain)), "explained_rows": "first 100 test patients plus patients A and B (102 rows)",
        "n_explained_positive": int(y_te[explain_pos].sum()),
        "max_abs_residual": float(k_resid.max()), "mean_abs_residual": float(k_resid.mean()), "tolerance": 1e-3,
        "nonzero_features_per_row": {"mean": float((np.abs(ksv.values) > 1e-12).sum(1).mean()),
                                     "max": int((np.abs(ksv.values) > 1e-12).sum(1).max())},
        "runtime_seconds": float(t_kernel),
    },
}

# --- Global rankings, direction patterns, grouped importance ---
group_of = {c: next((g for g in CATEGORICAL_COLS if c.startswith(g + "_")), c) for c in feat}


def pattern(v, s):
    v, s = np.asarray(v, float), np.asarray(s, float)
    rho = float(spearmanr(v, s).statistic) if v.std() > 0 and s.std() > 0 else None
    if set(np.unique(v)) <= {0.0, 1.0}:
        hi, lo, kind = v == 1, v == 0, "binary (present vs absent)"
    else:
        q25, q75 = np.percentile(v, [25, 75])
        hi, lo, kind = v >= q75, v <= q25, "continuous (top vs bottom quartile of value)"
    return {"kind": kind, "spearman_value_vs_shap": rho,
            "mean_shap_high_or_present": float(s[hi].mean()) if hi.any() else None, "n_high": int(hi.sum()),
            "mean_shap_low_or_absent": float(s[lo].mean()) if lo.any() else None, "n_low": int(lo.sum()),
            "share_positive_shap": float((s > 0).mean())}


def rank_block(values, Xrows, k=5):
    imp_s = pd.Series(np.abs(values).mean(0), index=feat).sort_values(ascending=False)
    return {"n_rows": int(len(values)), "all": {c: float(v) for c, v in imp_s.items()},
            "top5": [{"rank": i + 1, "feature": c, "mean_abs_shap": float(imp_s[c]),
                      "share_of_total": float(imp_s[c] / imp_s.sum()),
                      **pattern(Xrows[c].to_numpy(), values[:, feat.index(c)])} for i, c in enumerate(imp_s.index[:k])]}


def grouped_block(values, k=5):
    g = pd.DataFrame(values, columns=feat).T.groupby(group_of).sum().T
    m = g.abs().mean().sort_values(ascending=False)
    return {"rule": "signed SHAP summed over the one-hot columns of each original variable per case, then mean |.|",
            "top": [{"rank": i + 1, "variable": c, "mean_abs_shap": float(m[c]), "share_of_total": float(m[c] / m.sum())}
                    for i, c in enumerate(m.index[:k])], "all": {c: float(v) for c, v in m.items()}}


rf_vals_all = sv1.values
rf_vals_sub = sv1.values[explain_pos]
bmi_v = X_test["bmi"].to_numpy()
bmi_s = sv1[:, "bmi"].values
ev["global"] = {
    "rf_all_test": rank_block(rf_vals_all, X_test),
    "rf_same_102_rows": rank_block(rf_vals_sub, X_explain),
    "mlp_102_rows": rank_block(ksv.values, X_explain),
    "rf_grouped_all_test": grouped_block(rf_vals_all),
    "rf_grouped_102_rows": grouped_block(rf_vals_sub),
    "mlp_grouped_102_rows": grouped_block(ksv.values),
    "bmi_rf_all_test": {"share_recorded_bmi_negative_shap": float(np.mean(bmi_s[bmi_recorded] < 0)),
                        "mean_shap_bmi_lt_25": float(bmi_s[(bmi_v < 25) & bmi_recorded].mean()),
                        "mean_shap_bmi_25_30": float(bmi_s[(bmi_v >= 25) & (bmi_v < 30) & bmi_recorded].mean()),
                        "mean_shap_bmi_ge_30": float(bmi_s[(bmi_v >= 30) & bmi_recorded].mean())},
}


# --- Patient A and B tables ---
def local_block(values, base, xrow, score, k=3):
    s = pd.Series(values, index=feat)
    order = s.abs().sort_values(ascending=False).index
    top3 = [{"feature": c, "value": float(xrow[c]), "shap": float(s[c]),
             "direction": "increases" if s[c] > 0 else "decreases"} for c in order[:k]]
    return {"score": float(score), "base_value": float(base), "sum_shap": float(s.sum()),
            "top3": top3, "remaining_sum": float(s.drop(order[:k]).sum()), "n_remaining": int(len(s) - k),
            "all_features_sorted": [{"feature": c, "value": float(xrow[c]), "shap": float(s[c])} for c in order]}


ev["patients"] = {}
for nm, ip in (("A", A), ("B", B)):
    ev["patients"][nm] = {
        "test_position": int(ip), "dataframe_index": int(X_test.index[ip]),
        "actual": int(y_te[ip]),
        "raw_record": {k: (v if isinstance(v, str) else float(v)) for k, v in raw.loc[X_test.index[ip]].drop("id").items()},
        "bmi_recorded": bool(bmi_recorded[ip]),
        "rf": {"pred": int(rf_pred[ip]), "case_type": case_type(y_te[ip], rf_pred[ip]),
               **local_block(sv1[ip].values, sv1[ip].base_values, X_test.iloc[ip], rf_proba[ip])},
        "mlp": {"pred": int(mlp_pred[ip]), "case_type": case_type(y_te[ip], mlp_pred[ip]),
                **local_block(ksv.values[pos[ip]], k_base, X_test.iloc[ip], mlp_proba[ip])},
    }
ev["patients"]["A"]["selection_rule"] = ("highest-scoring true positive of the RF among test patients with a recorded BMI "
                                         "(xai_helpers.select_patients); chosen by rule, not by looks")
ev["patients"]["B"]["selection_rule"] = "lowest-scoring true negative of the RF among test patients with a recorded BMI"
ev["patients"]["A"]["mlp_values_for_rf_top3"] = {t["feature"]: float(ksv.values[pos[A]][feat.index(t["feature"])])
                                                 for t in ev["patients"]["A"]["rf"]["top3"]}


# --- Explanation stability ---
def kernel_run(background, seed, **kw):
    np.random.seed(seed)
    ke = shap.KernelExplainer(f, background)
    t0 = time.perf_counter()
    v = np.asarray(ke.shap_values(X_explain, nsamples=500, silent=True, **kw)).reshape(len(X_explain), len(feat))
    return v, float(np.ravel(ke.expected_value)[0]), time.perf_counter() - t0


WATCH = ["age", "heart_disease", "hypertension", "avg_glucose_level", "bmi", "gender_Male", "smoking_status_never smoked"]


def compare(main, other, ia):
    """main/other: (n_rows, n_features) SHAP arrays on the same rows; ia = row index of patient A."""
    ma, oa = pd.Series(main[ia], index=feat), pd.Series(other[ia], index=feat)
    mt3 = list(ma.abs().sort_values(ascending=False).index[:3])
    ot3 = list(oa.abs().sort_values(ascending=False).index[:3])
    mi, oi = pd.Series(np.abs(main).mean(0), index=feat), pd.Series(np.abs(other).mean(0), index=feat)
    m5, o5 = list(mi.sort_values(ascending=False).index[:5]), list(oi.sort_values(ascending=False).index[:5])
    big = np.abs(main) >= 0.02
    return {
        "patient_a_top3_main": mt3, "patient_a_top3_other": ot3, "patient_a_top3_overlap": len(set(mt3) & set(ot3)),
        "patient_a_sign_agreement_main_top3": {c: bool(np.sign(ma[c]) == np.sign(oa[c])) for c in mt3},
        "patient_a_shap_main_top3": {c: float(ma[c]) for c in mt3},
        "patient_a_shap_other_for_main_top3": {c: float(oa[c]) for c in mt3},
        "patient_a_spearman_all_features": float(spearmanr(ma, oa).statistic),
        "patient_a_max_abs_diff": float(np.abs(ma - oa).max()),
        "global_top5_main": m5, "global_top5_other": o5, "global_top5_overlap": len(set(m5) & set(o5)),
        "global_importance_spearman": float(spearmanr(mi, oi).statistic),
        "row_sign_agreement_where_main_abs_ge_0.02": float((np.sign(main)[big] == np.sign(other)[big]).mean()),
        "share_of_total_importance_other": {c: float(oi[c] / oi.sum()) for c in WATCH},
        "share_of_total_importance_main": {c: float(mi[c] / mi.sum()) for c in WATCH},
        "nonzero_features_per_row_other_mean": float((np.abs(other) > 1e-12).sum(1).mean()),
    }


ia = pos[A]
v_bg, base_bg, t_bg = kernel_run(shap.sample(X_train, 20, random_state=7), seed=7)
v_seed, base_seed, t_seed = kernel_run(shap.kmeans(X_train, 20), seed=12345)
v_l1, base_l1, t_l1 = kernel_run(shap.kmeans(X_train, 20), seed=2021, l1_reg=False)
rf2 = RandomForestClassifier(min_samples_leaf=20, random_state=7, n_jobs=-1).fit(X_train, y_train)
rf2_proba = rf2.predict_proba(X_test)[:, 1]
rf2_vals = shap.TreeExplainer(rf2)(X_test)[..., 1].values
ev["stability"] = {
    "mlp_different_background": {
        "what": "Kernel SHAP, background = 20 random training rows (shap.sample, seed 7) instead of k-means centroids; same rows, nsamples=500",
        "base_value": base_bg, "runtime_seconds": float(t_bg), **compare(ksv.values, v_bg, ia)},
    "mlp_different_sampling_seed": {
        "what": "Kernel SHAP, same k-means background, different sampling seed (12345)",
        "base_value": base_seed, "runtime_seconds": float(t_seed), **compare(ksv.values, v_seed, ia)},
    "mlp_no_l1_regularisation": {
        "what": "Kernel SHAP, same background and seed 2021, l1_reg=False (no 10-feature selection)",
        "base_value": base_l1, "runtime_seconds": float(t_l1),
        "max_abs_residual": float(np.abs(base_l1 + v_l1.sum(1) - k_out).max()), **compare(ksv.values, v_l1, ia)},
    "rf_second_seed": {
        "what": "RandomForest(min_samples_leaf=20, random_state=7) refit on the same training data, TreeSHAP on the full test set",
        "recall_stroke": float(((rf2_proba >= 0.5) & (y_te == 1)).sum() / y_te.sum()),
        "roc_auc": float(roc_auc_score(y_te, rf2_proba)), "patient_a_score_second_seed": float(rf2_proba[A]),
        "patient_a_score_main": float(rf_proba[A]),
        **compare(rf_vals_all, rf2_vals, A)},
}

# --- Environment ---
ev["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                     "shap": shap.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "imbalanced_learn": imblearn.__version__,
                     "matplotlib": matplotlib.__version__, "scipy": scipy.__version__,
                     "random_state": RANDOM_STATE, "generated": datetime.datetime.now().isoformat(timespec="seconds")}

(FIG / "report_evidence.json").write_text(json.dumps(ev, indent=2, default=jsonable))

# %%
m_ = ev["models"]
print("Held-out test, threshold 0.5")
print(pd.DataFrame({k: {x: m_[k][x] for x in ["accuracy", "balanced_accuracy", "precision_stroke", "recall_stroke",
                                               "f1_stroke", "roc_auc", "average_precision", "f1_macro"]}
                    for k in m_}).round(3).to_string())
for k in m_:
    print(k, m_[k]["confusion"], "Brier", round(m_[k]["calibration"]["brier"], 3),
          "mean score", round(m_[k]["calibration"]["mean_score"], 3))
sv_ = ev["shap_verification"]
print(f"RF residual (all test) {sv_['rf_treeshap']['max_abs_residual_all_test']:.2e}; "
      f"TreeSHAP {sv_['rf_treeshap']['runtime_seconds']:.1f}s; "
      f"Kernel residual {sv_['mlp_kernelshap']['max_abs_residual']:.2e}; "
      f"Kernel SHAP {sv_['mlp_kernelshap']['runtime_seconds']:.0f}s")
for k in ("rf_all_test", "mlp_102_rows"):
    print(k, [(r["feature"], round(r["mean_abs_shap"], 3),
               None if r["spearman_value_vs_shap"] is None else round(r["spearman_value_vs_shap"], 2))
              for r in ev["global"][k]["top5"]])
for k in ("rf_grouped_all_test", "mlp_grouped_102_rows"):
    print(k, [(r["variable"], round(r["mean_abs_shap"], 3)) for r in ev["global"][k]["top"]])
for nm in "AB":
    for mm in ("rf", "mlp"):
        p_ = ev["patients"][nm][mm]
        print(nm, mm, p_["case_type"], round(p_["score"], 3),
              [(t["feature"], round(t["shap"], 3)) for t in p_["top3"]], round(p_["remaining_sum"], 3))
KEYS = ("patient_a_top3_other", "patient_a_top3_overlap", "global_top5_overlap", "global_importance_spearman",
        "row_sign_agreement_where_main_abs_ge_0.02", "patient_a_spearman_all_features")
for k, v in ev["stability"].items():
    print(k, {x: v[x] for x in KEYS})
    print("   share (other) heart_disease/hypertension:",
          {c: round(v["share_of_total_importance_other"][c], 3) for c in ("heart_disease", "hypertension")})
print(json.dumps(ev["starter_script_run"], indent=1))
