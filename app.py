import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# Google Sheets Persistence
# -----------------------------
REQUIRED_COLUMNS = ["date", "rhr", "sleep_hours", "steps", "created_at"]

def gs_client():
    sa_info = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)

def open_worksheet():
    sheet_id = st.secrets["app"]["sheet_id"]
    ws_name = st.secrets["app"].get("worksheet_name", "observations")
    gc = gs_client()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(ws_name)
    return ws

def ensure_headers(ws):
    # Check first row; if empty, set headers
    first_row = ws.row_values(1)
    if not first_row:
        ws.append_row(REQUIRED_COLUMNS, value_input_option="RAW")
        return

    # If headers differ, best to stop to avoid corrupting data
    normalized = [c.strip().lower() for c in first_row]
    if normalized != REQUIRED_COLUMNS:
        raise ValueError(
            f"Worksheet headers must be exactly: {REQUIRED_COLUMNS}. "
            f"Found: {normalized}"
        )

def read_history_from_sheet(ws) -> pd.DataFrame:
    ensure_headers(ws)
    values = ws.get_all_records()  # list[dict]
    if not values:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(values)
    # Normalize types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["rhr"] = pd.to_numeric(df["rhr"], errors="coerce")

    # optional cols
    df["sleep_hours"] = pd.to_numeric(df.get("sleep_hours"), errors="coerce")
    df["steps"] = pd.to_numeric(df.get("steps"), errors="coerce")

    # keep only valid rows
    df = df.dropna(subset=["date", "rhr"])
    df = df.sort_values("date")
    return df

def upsert_rows(ws, df_new: pd.DataFrame) -> int:
    """
    Upsert by date (string YYYY-MM-DD). If date exists, update row; else append.
    """
    ensure_headers(ws)

    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"]).dt.date
    df_new["rhr"] = pd.to_numeric(df_new["rhr"], errors="coerce")
    if "sleep_hours" not in df_new.columns:
        df_new["sleep_hours"] = np.nan
    else:
        df_new["sleep_hours"] = pd.to_numeric(df_new["sleep_hours"], errors="coerce")

    if "steps" not in df_new.columns:
        df_new["steps"] = np.nan
    else:
        df_new["steps"] = pd.to_numeric(df_new["steps"], errors="coerce")

    df_new = df_new.dropna(subset=["date", "rhr"])
    if len(df_new) == 0:
        return 0

    # Fetch current sheet to find existing dates -> row indices
    existing = ws.get_all_values()  # includes header
    header = existing[0]
    rows = existing[1:]

    # Map date string -> row number in sheet (1-indexed)
    date_to_row = {}
    for i, r in enumerate(rows, start=2):
        if len(r) == 0:
            continue
        d = str(r[0]).strip()
        if d:
            date_to_row[d] = i

    now = datetime.utcnow().isoformat(timespec="seconds")

    updated = 0
    for _, r in df_new.iterrows():
        d_str = str(r["date"])
        row_data = [
            d_str,
            float(r["rhr"]),
            "" if pd.isna(r["sleep_hours"]) else float(r["sleep_hours"]),
            "" if pd.isna(r["steps"]) else int(r["steps"]),
            now,
        ]

        if d_str in date_to_row:
            row_idx = date_to_row[d_str]
            # Update the whole row range A:E
            ws.update(f"A{row_idx}:E{row_idx}", [row_data], value_input_option="RAW")
        else:
            ws.append_row(row_data, value_input_option="RAW")

        updated += 1

    return updated

def delete_day(ws, d: date):
    """
    Find row matching date and delete it.
    """
    ensure_headers(ws)
    all_vals = ws.get_all_values()
    rows = all_vals[1:]
    d_str = str(d)
    for i, r in enumerate(rows, start=2):
        if len(r) > 0 and str(r[0]).strip() == d_str:
            ws.delete_rows(i)
            return True
    return False


# -----------------------------
# Bayesian model (closed-form)
# y = w0 + w1*(sleep_centered) + w2*(steps_k_centered) + noise
# -----------------------------
@dataclass
class BayesResult:
    pred_mean: float
    pred_sd: float
    prob_high: float
    status: str
    message: str

def normal_cdf(x: float, mu: float, sd: float) -> float:
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sd * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def fit_posterior(X: np.ndarray, y: np.ndarray, m0: np.ndarray, V0: np.ndarray, obs_sd: float):
    s2 = obs_sd ** 2
    V0_inv = np.linalg.inv(V0)
    XtX = X.T @ X
    Vn_inv = V0_inv + (XtX / s2)
    Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ (V0_inv @ m0 + (X.T @ y) / s2)
    return mn, Vn

def make_design_matrix(df: pd.DataFrame, sleep_mean: float, stepsk_mean: float) -> np.ndarray:
    sleep = df["sleep_hours"].astype(float).to_numpy()
    steps = df["steps"].astype(float).to_numpy()
    steps_k = steps / 1000.0

    sleep_c = np.where(np.isnan(sleep), 0.0, sleep - sleep_mean)
    stepsk_c = np.where(np.isnan(steps_k), 0.0, steps_k - stepsk_mean)

    return np.column_stack([np.ones(len(df)), sleep_c, stepsk_c])

def score_day(history: pd.DataFrame, target_day: date, obs_sd: float = 3.0) -> BayesResult:
    row = history[history["date"] == target_day]
    if row.empty:
        return BayesResult(float("nan"), float("nan"), float("nan"), "N/A", f"No record found for {target_day}.")

    y_today = float(row.iloc[0]["rhr"])
    sleep_today = row.iloc[0].get("sleep_hours", np.nan)
    steps_today = row.iloc[0].get("steps", np.nan)

    past = history[history["date"] < target_day].copy()

    # Priors (MVP-friendly, stable)
    m0 = np.array([65.0, -1.0, 0.5])         # baseline, sleep effect, steps(k) effect
    V0 = np.diag([10.0**2, 0.7**2, 0.5**2])

    if len(past) >= 2:
        sleep_mean = float(np.nanmean(past["sleep_hours"])) if past["sleep_hours"].notna().any() else 7.0
        stepsk_mean = float(np.nanmean(past["steps"]) / 1000.0) if past["steps"].notna().any() else 7.0
        X = make_design_matrix(past, sleep_mean=sleep_mean, stepsk_mean=stepsk_mean)
        y = past["rhr"].astype(float).to_numpy()
        mn, Vn = fit_posterior(X, y, m0, V0, obs_sd=obs_sd)
    else:
        sleep_mean, stepsk_mean = 7.0, 7.0
        mn, Vn = m0, V0

    df_today = pd.DataFrame([{
        "sleep_hours": (np.nan if pd.isna(sleep_today) else float(sleep_today)),
        "steps": (np.nan if pd.isna(steps_today) else float(steps_today)),
    }])
    x = make_design_matrix(df_today, sleep_mean=sleep_mean, stepsk_mean=stepsk_mean)[0]

    pred_mean = float(x @ mn)
    pred_var = float((obs_sd**2) + (x @ Vn @ x))
    pred_sd = math.sqrt(max(pred_var, 1e-9))

    prob_high = normal_cdf(y_today, pred_mean, pred_sd)

    if prob_high >= 0.90:
        status = "🔴 Unusual (elevated)"
        msg = (
            f"RHR {y_today:.1f} is unusually high for you (≈{prob_high*100:.0f}th percentile), "
            f"after accounting for sleep/steps if provided.\n\n"
            f"Expected today: {pred_mean:.1f} ± {pred_sd:.1f} bpm."
        )
    elif prob_high >= 0.70:
        status = "🟡 Elevated"
        msg = (
            f"RHR {y_today:.1f} looks somewhat elevated (≈{prob_high*100:.0f}th percentile).\n\n"
            f"Expected today: {pred_mean:.1f} ± {pred_sd:.1f} bpm."
        )
    else:
        status = "🟢 Within normal"
        msg = (
            f"RHR {y_today:.1f} looks within your normal range (≈{prob_high*100:.0f}th percentile).\n\n"
            f"Expected today: {pred_mean:.1f} ± {pred_sd:.1f} bpm."
        )

    return BayesResult(pred_mean, pred_sd, prob_high, status, msg)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Personal Baseline Monitor (Sheets)", layout="wide")
st.title("Personal Baseline Monitor (Google Sheets + Bayesian MVP)")
st.caption("Add observations via UI or CSV. Data is stored in Google Sheets and used to generate a message for a selected day.")

# Connect once; cache worksheet handle to reduce API calls
@st.cache_resource
def get_ws():
    ws = open_worksheet()
    ensure_headers(ws)
    return ws

ws = get_ws()

@st.cache_data(ttl=15)
def load_history_cached():
    return read_history_from_sheet(ws)

history = load_history_cached()

with st.sidebar:
    st.header("Add data")

    tab_form, tab_csv = st.tabs(["➕ Manual entry", "⬆️ CSV upload"])

    with tab_form:
        with st.form("manual_entry", clear_on_submit=False):
            default_day = date.today() - timedelta(days=1)
            d = st.date_input("Date", value=default_day)
            rhr = st.number_input("Resting heart rate (bpm)", min_value=30.0, max_value=140.0, value=62.0, step=0.5)

            sleep = st.number_input("Sleep hours (optional)", min_value=0.0, max_value=16.0, value=0.0, step=0.25)
            steps = st.number_input("Steps (optional)", min_value=0, max_value=100000, value=0, step=100)

            sleep_val = np.nan if sleep == 0.0 else float(sleep)
            steps_val = np.nan if steps == 0 else int(steps)

            submitted = st.form_submit_button("Save / Update day")
            if submitted:
                df_in = pd.DataFrame([{
                    "date": d,
                    "rhr": float(rhr),
                    "sleep_hours": sleep_val,
                    "steps": steps_val,
                }])
                n = upsert_rows(ws, df_in)
                st.success(f"Saved/updated {n} row(s).")
                st.cache_data.clear()

    with tab_csv:
        st.write("CSV must include: `date,rhr` (optional: `sleep_hours,steps`).")
        st.code("date,rhr,sleep_hours,steps\n2026-01-14,62,7.5,8200", language="text")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                up = pd.read_csv(uploaded)
                up.columns = [c.strip().lower() for c in up.columns]
                if not {"date", "rhr"}.issubset(up.columns):
                    st.error("CSV must include at least: date, rhr")
                else:
                    if "sleep_hours" not in up.columns:
                        up["sleep_hours"] = np.nan
                    if "steps" not in up.columns:
                        up["steps"] = np.nan

                    n = upsert_rows(ws, up[["date", "rhr", "sleep_hours", "steps"]])
                    st.success(f"Saved/updated {n} row(s).")
                    st.cache_data.clear()
            except Exception as e:
                st.exception(e)

    st.divider()
    st.caption("Tip: re-entering the same date updates that row (upsert).")


left, right = st.columns([1.1, 1])

with left:
    st.subheader("Daily message")

    if history is None or len(history) == 0:
        st.info("No history yet. Add a day using the form or upload a CSV.")
        st.stop()

    history_sorted = history.sort_values("date")
    min_d = history_sorted["date"].min()
    max_d = history_sorted["date"].max()

    target_day = st.date_input("Select day", value=max_d, min_value=min_d, max_value=max_d)
    obs_sd = st.slider("Assumed wearable noise (sd, bpm)", 1.0, 6.0, 3.0, 0.5)

    res = score_day(history_sorted, target_day=target_day, obs_sd=obs_sd)
    st.markdown(f"### {res.status}")
    st.write(res.message)

    r = history_sorted[history_sorted["date"] == target_day].iloc[0]
    st.caption(
        f"Inputs: RHR={float(r['rhr']):.1f}, "
        f"sleep_hours={'—' if pd.isna(r['sleep_hours']) else float(r['sleep_hours']):.2f}, "
        f"steps={'—' if pd.isna(r['steps']) else int(r['steps'])}"
    )

    with st.expander("Manage this day"):
        if st.button("Delete selected day"):
            ok = delete_day(ws, target_day)
            if ok:
                st.success(f"Deleted {target_day}.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Could not find that date in the sheet (nothing deleted).")

with right:
    st.subheader("History view")
    plot_df = history.sort_values("date").copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    st.line_chart(plot_df.set_index("date")[["rhr"]])

    st.dataframe(history.sort_values("date", ascending=False), use_container_width=True)

st.caption("Data persistence is handled by Google Sheets (service account).")
