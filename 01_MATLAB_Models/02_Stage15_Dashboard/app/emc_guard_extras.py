"""
EMC-GUARD Stage 15 add-ons. Does NOT retrain or modify the frozen models.

Wire-up (3 places marked ADAPT):
  - FEATURES must match the column order your frozen models were trained on.
  - Pass your loaded models as {"cap": model_C, "ind": model_L}.
  - Pass your 216-row dataframe (inputs + MATLAB targets) as `df`.
"""
import numpy as np
import pandas as pd

# ---------- 1. Supported domain (from the frozen 216-point grid) ----------
LEVELS = {
    "PairSeparation_mm": [8, 12, 16, 20, 24, 28],
    "PairPitch_mm": [2, 3, 4],
    "Overlap_mm": [150, 300, 450],
    "GroundClearance_mm": [10, 25, 50, 100],
}
FEATURES = list(LEVELS)  # ADAPT: must equal the training column order
TARGETS = {"cap": "TotalCapacitance_pF_per_m", "ind": "AbsMutualInductance_nH"}


def domain_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Input": list(LEVELS),
            "Min": [min(v) for v in LEVELS.values()],
            "Max": [max(v) for v in LEVELS.values()],
            "Supported levels": [", ".join(map(str, v)) for v in LEVELS.values()],
        }
    )


# ---------- 2. Strict domain check ----------
def classify_input(cfg: dict) -> tuple[str, str]:
    """Return (status, message). status in {'on-grid','interpolated','out-of-domain'}."""
    missing = [k for k in FEATURES if k not in cfg]
    if missing:
        return "out-of-domain", f"Missing inputs: {', '.join(missing)}. Prediction rejected."
    outside = [
        k for k, v in LEVELS.items() if not (min(v) <= cfg[k] <= max(v))
    ]
    if outside:
        return "out-of-domain", f"Outside design ranges for: {', '.join(outside)}. Prediction rejected."
    off_grid = [k for k, v in LEVELS.items() if cfg[k] not in v]
    if off_grid:
        return (
            "interpolated",
            f"Between supported levels for: {', '.join(off_grid)}. The frozen predictor does not "
            "support this; it would need a separately validated interpolation pathway. Prediction rejected.",
        )
    return "on-grid", "Exact match to a design-grid configuration; AI vs MATLAB agreement was checked."


def safe_predict(models: dict, cfg: dict):
    status, msg = classify_input(cfg)
    if status != "on-grid":
        return status, msg, None
    X = pd.DataFrame([[cfg[k] for k in FEATURES]], columns=FEATURES)
    out = {t: float(models[t].predict(X)[0]) for t in TARGETS}  # ADAPT if models need other input
    return status, msg, out


# ---------- 3. Model disagreement (NOT a confidence interval) ----------
def _forest(model):
    # supports a bare ExtraTreesRegressor or an sklearn Pipeline ending in one
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return est


def model_disagreement(model, cfg: dict) -> dict:
    X = pd.DataFrame([[cfg[k] for k in FEATURES]], columns=FEATURES)
    Xt = model[:-1].transform(X) if hasattr(model, "steps") else X.values
    preds = np.array([t.predict(Xt)[0] for t in _forest(model).estimators_])
    return {"mean": preds.mean(), "std": preds.std(ddof=1), "min": preds.min(), "max": preds.max()}


# ---------- 4. User-weighted two-objective Pareto ----------
def pareto_weighted(df: pd.DataFrame, w_cap: float = 0.5, cap_col=None, ind_col=None) -> pd.DataFrame:
    """Both objectives are minimised. df must hold one row per on-grid config with
    the objective columns (use AI predictions or MATLAB reference, label which in the UI)."""
    cap_col = cap_col or TARGETS["cap"]
    ind_col = ind_col or TARGETS["ind"]
    d = df.copy()
    for c in (cap_col, ind_col):
        rng = d[c].max() - d[c].min()
        d[c + "_norm"] = (d[c] - d[c].min()) / (rng if rng else 1.0)
    d["weighted_score"] = w_cap * d[cap_col + "_norm"] + (1 - w_cap) * d[ind_col + "_norm"]
    a, b = d[cap_col].values, d[ind_col].values
    d["pareto_optimal"] = [
        not np.any((a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))) for i in range(len(d))
    ]
    return d.sort_values("weighted_score").reset_index(drop=True)


# ---------- 5. Per-output feature importance (descriptive, on the design grid) ----------
def permutation_table(models: dict, df: pd.DataFrame, n_repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    rows = []
    X = df[FEATURES]
    for t, col in TARGETS.items():
        r = permutation_importance(
            models[t], X, df[col], n_repeats=n_repeats, random_state=seed, scoring="r2"
        )
        for f, m, s in zip(FEATURES, r.importances_mean, r.importances_std):
            rows.append({"Output": col, "Input": f, "Importance (drop in R²)": m, "Std": s})
    return pd.DataFrame(rows)


# ---------- 6. Fixed wording ----------
LIMITATIONS = (
    "Reproduction of MATLAB is verified on the original 216-configuration design grid. "
    "Leave-one-geometry-level-out tests showed weaker prediction on unseen geometry levels, "
    "so unseen levels are rejected rather than predicted. Tree spread is model disagreement, "
    "not a confidence interval. Physical validation and victim-voltage prediction are future work "
    "(victim voltage needs a validated circuit model and additional parameters)."
)
SCOPE_NOTE = "Pareto and sensitivity views are screening aids for L and C only, not EMI-reduction claims."


# ---------- Streamlit renderers (call from your pages) ----------
def render_domain_panel(st):
    st.subheader("Supported input domain")
    st.dataframe(domain_table(), hide_index=True)
    st.caption("Only exact grid combinations are predicted. 6 × 3 × 3 × 4 = 216 configurations.")


def render_predict_panel(st, models, key_prefix="p"):
    cfg = {k: st.selectbox(k, LEVELS[k], key=f"{key_prefix}_{k}") for k in FEATURES}
    # Optional free entry to demonstrate the rejection logic:
    with st.expander("Test the domain check with arbitrary values"):
        cfg = {k: st.number_input(k, value=float(cfg[k]), key=f"{key_prefix}_n_{k}") for k in FEATURES}
    status, msg, out = safe_predict(models, cfg)
    {"on-grid": st.success, "interpolated": st.warning, "out-of-domain": st.error}[status](
        f"{status.upper()}: {msg}"
    )
    if out:
        cols = st.columns(len(TARGETS))
        for c, (t, col) in zip(cols, TARGETS.items()):
            d = model_disagreement(models[t], cfg)
            c.metric(col, f"{out[t]:.4g}")
            c.caption(f"Model disagreement (tree std): {d['std']:.3g}. Not a confidence interval.")


def render_pareto_panel(st, df):
    w = st.slider("Weight on capacitance (rest goes to mutual inductance)", 0.0, 1.0, 0.5, 0.05)
    res = pareto_weighted(df, w)
    st.dataframe(res.head(10))
    st.caption(SCOPE_NOTE)
    return res


def render_limitations(st):
    st.warning(LIMITATIONS)
