"""Streamlit UI for the Haor Flash Flood Prediction System.

Run:  python -m streamlit run app.py
(Train first with: python train.py)
"""
import json

import joblib
import pandas as pd
import streamlit as st

import common

st.set_page_config(page_title="Haor Flash Flood Predictor", page_icon="🌊", layout="wide")


# ---------- cached loaders ----------
@st.cache_resource
def load_model(name: str):
    return joblib.load(common.model_path(name))["model"]


@st.cache_data
def load_data():
    raw = common.load_raw()
    X, y = common.make_xy(raw)
    return raw, X, y


if not all(common.model_path(n).exists() for n in common.MODEL_NAMES):
    st.error("No trained models found. Run `python train.py` first, then reload this page.")
    st.stop()

raw, X, y = load_data()
INPUTS = [f for f in common.FEATURES if f != "vv_vh_ratio"]  # ratio is computed

# ---------- session state: defaults = dataset medians ----------
for f in INPUTS:
    st.session_state.setdefault(f"in_{f}", float(X[f].median()))
st.session_state.setdefault("ndwi_ok", True)


def load_sample():
    """Callback: copy a historical event into the form fields."""
    choice = st.session_state["sample_choice"]
    if choice == "(enter values manually)":
        return
    idx = raw.index[raw["event_id"] == choice.split(" | ")[0]][0]
    for f in INPUTS:
        val = X.loc[idx, f]
        if pd.isna(val):
            if f == "ndwi":
                st.session_state["ndwi_ok"] = False
            continue
        st.session_state[f"in_{f}"] = float(val)
        if f == "ndwi":
            st.session_state["ndwi_ok"] = True


# ---------- sidebar ----------
with st.sidebar:
    st.header("🌊 Haor Flash Flood Predictor")
    st.caption("Sunamganj / Sylhet haor wetlands, NE Bangladesh")
    model_name = st.radio("Model", common.MODEL_NAMES,
                          help="Both models are trained on the same 77 events.")
    options = ["(enter values manually)"] + [
        f"{r.event_id} | {r.date} | {'FLOOD' if r.flood_label else 'no flood'}"
        for r in raw.itertuples()
    ]
    st.selectbox("Load a historical event", options, key="sample_choice",
                 on_change=load_sample,
                 help="Fills the form with a real event so you can see how the model responds.")
    st.divider()
    st.caption("Educational prototype. Not an official warning system.")

model = load_model(model_name)

st.title("Flash Flood Prediction System — Haor Region")

tab_predict, tab_perf, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model performance", "🗂️ Data explorer", "ℹ️ About"])

# ---------- tab 1: predict ----------
with tab_predict:
    st.write("Enter the conditions for a date, or load a historical event from the sidebar.")
    left, right = st.columns([3, 2], gap="large")

    def num(feature, lo, hi, step, disabled=False):
        label, hint = common.FEATURE_INFO[feature]
        return st.number_input(label, min_value=lo, max_value=hi, step=step,
                               key=f"in_{feature}", help=hint or None,
                               disabled=disabled)

    with left:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Radar & water")
            num("VV", -40.0, 0.0, 0.1)
            num("VH", -45.0, -5.0, 0.1)
            num("upstream_vv", -30.0, 0.0, 0.1)
            ndwi_ok = st.checkbox("NDWI available (cloud-free image)", key="ndwi_ok")
            num("ndwi", -1.0, 1.0, 0.01, disabled=not ndwi_ok)
        with c2:
            st.subheader("Rainfall")
            num("rainfall", 0.0, 500.0, 0.5)
            num("rain_24h", 0.0, 500.0, 0.5)
            num("rain_96h", 0.0, 1000.0, 1.0)
        with c3:
            st.subheader("Weather & soil")
            num("soil_moisture", 0.0, 100.0, 0.1)
            num("temp", 0.0, 50.0, 0.1)
            num("temp_anomaly", -10.0, 10.0, 0.1)
            num("wind", 0.0, 40.0, 0.1)

    values = {f: st.session_state[f"in_{f}"] for f in INPUTS}
    row = common.row_from_inputs(values, ndwi_available=ndwi_ok)
    prob = float(model.predict_proba(row)[0, 1])
    level = common.risk_level(prob)

    with right:
        st.subheader(f"Prediction ({model_name})")
        st.metric("Flash flood probability", f"{prob:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        if level == "High":
            st.error("🔴 **HIGH risk** — conditions resemble past flash flood events.")
        elif level == "Moderate":
            st.warning("🟠 **MODERATE risk** — mixed signals; monitor closely.")
        else:
            st.success("🟢 **LOW risk** — conditions resemble non-flood events.")
        st.caption("Thresholds: Low < 35%, Moderate 35–65%, High > 65%. "
                   "These are illustrative, not calibrated for operations.")

        # show both models side by side
        other = [n for n in common.MODEL_NAMES if n != model_name][0]
        p_other = float(load_model(other).predict_proba(row)[0, 1])
        st.caption(f"For comparison, {other} gives {p_other:.0%}.")

        st.markdown("**How your inputs compare with past events**")
        cmp_df = pd.DataFrame({
            "Your input": row.iloc[0][INPUTS],
            "Avg. non-flood": X[y == 0][INPUTS].mean(),
            "Avg. flood": X[y == 1][INPUTS].mean(),
        }).round(2)
        cmp_df.index = [common.FEATURE_INFO[f][0] for f in INPUTS]
        st.dataframe(cmp_df, height=330)

# ---------- tab 2: performance ----------
with tab_perf:
    if not common.METRICS_PATH.exists():
        st.info("Run `python train.py` to generate metrics.")
    else:
        m = json.loads(common.METRICS_PATH.read_text())[model_name]
        st.subheader(f"{model_name} — cross-validated performance")
        st.caption(f"{m['n_events']} events ({m['n_flood']} flood). {m['cv']}. "
                   "± shows the spread between folds; with this little data it is large.")
        c = st.columns(5)
        for col, (k, label) in zip(c, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                       ("recall", "Recall"), ("f1", "F1-score")]):
            col.metric(label, f"{m[k]:.3f}", f"± {m[k + '_std']:.3f}", delta_color="off")
        c[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Confusion matrix** (out-of-fold predictions)")
            cm = m["confusion"]
            st.dataframe(pd.DataFrame(
                [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                index=["Actual: no flood", "Actual: flood"],
                columns=["Predicted: no flood", "Predicted: flood"]))
            st.caption("For early warning, **recall** (catching real floods) usually matters "
                       "more than precision; a missed flood costs more than a false alarm.")
        with right:
            st.markdown("**Feature importance** (% share)")
            if common.IMPORTANCE_PATH.exists():
                imp = pd.read_csv(common.IMPORTANCE_PATH, index_col=0)
                st.bar_chart(imp[model_name])
                if model_name == "Logistic Regression":
                    st.caption("Based on absolute standardised coefficients.")

        st.markdown("**Model comparison**")
        if common.CV_RESULTS_PATH.exists():
            res = pd.read_csv(common.CV_RESULTS_PATH)
            st.dataframe(res.round(3), hide_index=True)
            st.caption("Differences between the two models are smaller than the fold-to-fold "
                       "spread, so neither is clearly better on 77 events.")

# ---------- tab 3: data explorer ----------
with tab_data:
    st.subheader("Haor flash flood events (2014–2024)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Events", len(raw))
    c2.metric("Flood events", int(y.sum()))
    c3.metric("Non-flood events", int((1 - y).sum()))

    plot_df = X.copy()
    plot_df["Outcome"] = y.map({1: "Flood", 0: "No flood"})
    a, b = st.columns(2)
    xf = a.selectbox("X axis", INPUTS, index=INPUTS.index("rain_96h"))
    yf = b.selectbox("Y axis", INPUTS, index=INPUTS.index("VV"))
    st.scatter_chart(plot_df, x=xf, y=yf, color="Outcome")
    with st.expander("Show raw data"):
        st.dataframe(raw)

# ---------- tab 4: about ----------
with tab_about:
    st.markdown("""
### About this project
**Goal:** estimate whether a flash flood is occurring in the haor wetlands of Sunamganj/Sylhet
(NE Bangladesh) from satellite and reanalysis variables.

**Data:** *SAR-verified flash flood event dataset for the haor wetlands of northeastern
Bangladesh (2014–2024)* from Mendeley Data. 77 events with a binary flood label and
14 features (Sentinel-1 radar, Sentinel-2 NDWI, ERA5-Land rainfall, soil moisture, temperature, wind).

**Method:** Logistic Regression and Random Forest, evaluated with 5-fold stratified
cross-validation repeated 10 times. Missing NDWI values are filled with the median
inside each pipeline.

**Data preparation:** `source` is dropped because it leaks the label; `slope` and `twi`
are constant; `vv_vh_ratio` is recomputed as VV / VH.

### Limitations
- Only **77 events** from a single area, so results are noisy and may not transfer elsewhere.
- Inputs include satellite radar features, so this is a *nowcasting* tool for an event date,
  not a forecast of future floods.
- The labels come from reports and literature; they are not ground-truth gauge readings.
- Risk thresholds are illustrative. **Do not use for real warnings or decisions.**
""")
