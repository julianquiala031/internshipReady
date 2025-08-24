# app.py — Water Quality Dashboard (Streamlit)
# -------------------------------------------------------------
# Features
# - Upload CSV (or auto-generate a realistic demo dataset)
# - Smart column detection (time, lat, lon, station)
# - Sidebar filters: date range, station, parameter, rolling window, outlier removal
# - KPI summary cards
# - Interactive map (OpenStreetMap, no token needed)
# - Time series with optional rolling mean
# - Box/violin plots, scatter w/ trendline, correlation heatmap
# - Basic threshold alerts (DO, pH, turbidity, chlorophyll, temperature)
# - Anomaly flags (z-score based)
# - Download filtered data
# -------------------------------------------------------------

import io
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Water Quality Dashboard",
    page_icon="💧",
    layout="wide",
)

# --------------------------
# Utilities & Config
# --------------------------

NUMERIC_PARAM_CANDIDATES = [
    ("temperature", "°C"),
    ("salinity", "psu"),
    ("dissolved_oxygen", "mg/L"),
    ("do", "mg/L"),
    ("oxygen", "mg/L"),
    ("ph", "pH"),
    ("turbidity", "NTU"),
    ("chlorophyll", "µg/L"),
    ("chl", "µg/L"),
    ("conductivity", "µS/cm"),
]

DEFAULT_THRESHOLDS = {
    # Basic, non-regulatory heuristics for quick alerts
    "dissolved_oxygen": {"min": 5},          # mg/L
    "do": {"min": 5},                        # alias
    "ph": {"min": 6.5, "max": 8.5},
    "turbidity": {"max": 50},               # NTU
    "chlorophyll": {"max": 20},             # µg/L
    "temperature": {"max": 32},             # °C
}

@st.cache_data(show_spinner=False)
def generate_demo_data(n_days: int = 7, freq_min: int = 30, n_stations: int = 4) -> pd.DataFrame:
    """Create a synthetic but realistic dataset for quick demo/testing."""
    rng = pd.date_range(
        end=pd.Timestamp.utcnow().floor("min"),
        periods=(n_days * 24 * 60) // freq_min,
        freq=f"{freq_min}min",
        tz="UTC",
    )
    base_lat, base_lon = 25.7617, -80.1918  # Miami-ish
    stations = [f"STN-{i+1:02d}" for i in range(n_stations)]

    rows = []
    for stn in stations:
        lat_jitter = np.random.uniform(-0.15, 0.15)
        lon_jitter = np.random.uniform(-0.15, 0.15)
        lat = base_lat + lat_jitter
        lon = base_lon + lon_jitter

        # Seasonal-ish variation
        temp = 27 + 3 * np.sin(np.linspace(0, 6 * np.pi, len(rng))) + np.random.normal(0, 0.6, len(rng))
        sal = 35 + 1.0 * np.sin(np.linspace(0, 3 * np.pi, len(rng))) + np.random.normal(0, 0.3, len(rng))
        do = 6.5 + 0.8 * np.cos(np.linspace(0, 4 * np.pi, len(rng))) + np.random.normal(0, 0.25, len(rng))
        ph = 7.9 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, len(rng))) + np.random.normal(0, 0.03, len(rng))
        chl = 10 + 4 * np.maximum(0, np.sin(np.linspace(0, 5 * np.pi, len(rng)))) + np.random.normal(0, 1.2, len(rng))
        turb = 8 + 6 * np.maximum(0, np.cos(np.linspace(0, 5 * np.pi, len(rng)))) + np.random.normal(0, 1.5, len(rng))

        df_stn = pd.DataFrame({
            "timestamp": rng,
            "station": stn,
            "latitude": lat,
            "longitude": lon,
            "temperature": temp,
            "salinity": sal,
            "dissolved_oxygen": do,
            "ph": ph,
            "chlorophyll": chl,
            "turbidity": turb,
        })
        rows.append(df_stn)

    df = pd.concat(rows, ignore_index=True)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + strip spaces/units from column names for easier matching."""
    mapping = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=mapping)
    return df


def detect_columns(df: pd.DataFrame):
    """Detect standard columns and candidate numeric parameters."""
    cols = {c: c.lower() for c in df.columns}

    # Time
    time_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["time", "date"]):
            time_col = c
            break
    if time_col is None and "timestamp" in cols.values():
        time_col = [k for k, v in cols.items() if v == "timestamp"][0]

    # Latitude/Longitude
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if "lat" in lc and lat_col is None:
            lat_col = c
        if ("lon" in lc or "lng" in lc or "long" in lc) and lon_col is None:
            lon_col = c

    # Station name/id
    station_col = None
    for key in ["station", "site", "location", "station_id", "site_id", "stationname", "sitename"]:
        candidates = [c for c in df.columns if c.lower() == key]
        if candidates:
            station_col = candidates[0]
            break

    # Numeric parameters
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Some params may be numeric strings; try to coerce simple ones later
    candidates = []
    for name, unit in NUMERIC_PARAM_CANDIDATES:
        for c in df.columns:
            if name == c.lower():
                candidates.append((c, unit))
    # Include other numeric columns that aren't lat/lon
    for c in numeric_cols:
        if c not in [lat_col, lon_col] and c != (station_col or ""):
            if c not in [x[0] for x in candidates]:
                candidates.append((c, ""))

    return {
        "time": time_col,
        "lat": lat_col,
        "lon": lon_col,
        "station": station_col,
        "params": candidates,
    }


def parse_times(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col])
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def zscore_anomalies(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    if s.empty:
        return pd.Series([], dtype=bool)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([False] * len(s), index=s.index)
    return (np.abs((s - mean) / std) > threshold)


def apply_iqr_filter(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.between(lower, upper) | series.isna()


# --------------------------
# Data Ingestion
# --------------------------

st.title("💧 Water Quality Dashboard")
st.caption("Upload your CSV or explore the demo dataset. Columns are auto-detected.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])  
    use_demo = st.toggle("Use demo dataset", value=uploaded is None)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif use_demo:
    df_raw = generate_demo_data(n_days=10, freq_min=30, n_stations=5)
else:
    st.info("Upload a CSV or toggle the demo dataset to continue.")
    st.stop()

# Normalize & detect
_df = _normalize_columns(df_raw)
meta = detect_columns(_df)

# Required columns (best effort)
if not meta["time"]:
    st.error("Couldn't detect a time column. Include a column like 'timestamp', 'time', or 'date'.")
    st.stop()

time_col = meta["time"]
lat_col = meta["lat"] or "latitude"
lon_col = meta["lon"] or "longitude"
station_col = meta["station"] or "station"

# Ensure columns exist; if not, try to create from demo
if lat_col not in _df.columns:
    # fallback: add default lat
    _df[lat_col] = 25.75
if lon_col not in _df.columns:
    _df[lon_col] = -80.20
if station_col not in _df.columns:
    _df[station_col] = "STN-00"

_df = parse_times(_df, time_col)

# Parameter options
param_candidates = [p for p, _u in meta["params"] if p not in [lat_col, lon_col, station_col, time_col]]
if not param_candidates:
    st.error("No numeric parameters detected. Include columns like temperature, salinity, DO, pH, etc.")
    st.stop()

# Try coercing params to numeric (idempotent if already numeric)
_df = coerce_numeric(_df, param_candidates)

# Sidebar Filters
with st.sidebar:
    st.header("Filters")

    # Date range
    tmin = _df[time_col].min()
    tmax = _df[time_col].max()
    if pd.isna(tmin) or pd.isna(tmax):
        st.error("Invalid or empty time values after parsing.")
        st.stop()
    start, end = st.date_input(
        "Date range",
        value=(tmin.date(), tmax.date()),
        min_value=tmin.date(),
        max_value=tmax.date(),
    )

    # Station
    stations = sorted(_df[station_col].astype(str).unique())
    station_sel = st.multiselect("Stations", stations, default=stations)

    # Parameter selection
    param_display_map = {}
    for p, u in meta["params"]:
        label = f"{p} ({u})" if u else p
        param_display_map[label] = p
    params_selected = st.multiselect(
        "Parameters",
        list(param_display_map.keys()),
        default=[k for k in param_display_map.keys() if any(x in k.lower() for x in ["temperature", "salinity", "oxygen", "ph", "chlorophyll", "turbidity"])][:6],
    )
    params = [param_display_map[k] for k in params_selected] if params_selected else param_candidates[:4]

    # Rolling window for smoothing
    rolling = st.slider("Rolling window (samples)", 1, 24, 6)

    # Outlier removal
    apply_outliers = st.checkbox("Remove outliers (IQR method)", value=False)

    # Anomaly threshold (z-score)
    z_thr = st.slider("Anomaly threshold (z-score)", 2.0, 5.0, 3.0, 0.1)

# Apply filters
mask = (
    (_df[time_col] >= pd.Timestamp(start).tz_localize("UTC")) &
    (_df[time_col] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) &
    (_df[station_col].astype(str).isin(station_sel))
)
df = _df.loc[mask].copy()

# IQR outlier filtering (per parameter)
if apply_outliers:
    for p in params:
        if p in df.columns:
            keep = apply_iqr_filter(df[p])
            df.loc[~keep, p] = np.nan

if df.empty:
    st.warning("No data after applying filters.")
    st.stop()

# --------------------------
# KPI Row
# --------------------------

def kpi_card(label: str, value: float | str, help_text: str = ""):
    st.metric(label, value)
    if help_text:
        st.caption(help_text)

kpi_cols = st.columns(min(4, len(params)))
for i, p in enumerate(params[:4]):
    col = kpi_cols[i]
    with col:
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            kpi_card(p.title(), f"{df[p].mean():.2f}", help_text="Mean over filtered data")
        else:
            kpi_card(p.title(), "—")

# --------------------------
# Map
# --------------------------

st.subheader("Map of Measurements")

# Aggregate to latest per station for mapping clarity
latest_idx = df.groupby(station_col)[time_col].idxmax()
latest = df.loc[latest_idx, [station_col, lat_col, lon_col] + params].dropna(subset=[lat_col, lon_col])

# Build hover info using first selected parameter as color
color_param = params[0] if params else None

if not latest.empty:
    fig_map = px.scatter_mapbox(
        latest,
        lat=lat_col,
        lon=lon_col,
        color=color_param if color_param in latest.columns else None,
        hover_data=[station_col] + [p for p in params if p in latest.columns],
        zoom=9,
        height=420,
        color_continuous_scale="Viridis",
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No geolocated data available to render the map.")

# --------------------------
# Time Series
# --------------------------

st.subheader("Time Series")

tabs = st.tabs([p for p in params])
for tab, p in zip(tabs, params):
    with tab:
        if p not in df.columns:
            st.info(f"Parameter '{p}' not found in data.")
            continue
        df_plot = df[[time_col, station_col, p]].dropna()
        if df_plot.empty:
            st.info("No data to plot.")
            continue
        # Rolling mean per station
        df_plot = df_plot.sort_values(time_col)
        df_plot[f"{p}_roll"] = df_plot.groupby(station_col)[p].transform(lambda s: s.rolling(rolling, min_periods=1).mean())

        fig_ts = go.Figure()
        for stn, g in df_plot.groupby(station_col):
            fig_ts.add_trace(go.Scatter(
                x=g[time_col], y=g[p], mode="lines", name=f"{stn} — raw", hovertemplate="%{y:.2f}<extra>"+str(stn)+"</extra>")
            )
            fig_ts.add_trace(go.Scatter(
                x=g[time_col], y=g[f"{p}_roll"], mode="lines", name=f"{stn} — {rolling}pt avg", line=dict(dash="dash")))

        fig_ts.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Time",
            yaxis_title=p,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# --------------------------
# Distribution & Relationships
# --------------------------

st.subheader("Distributions & Relationships")
colA, colB = st.columns(2)

with colA:
    if len(params) >= 1:
        p = params[0]
        df_box = df[[station_col, p]].dropna()
        if not df_box.empty:
            fig_box = px.box(df_box, x=station_col, y=p, points="all", title=f"{p} by Station")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Not enough data for box plot.")

with colB:
    if len(params) >= 2:
        x, y = params[:2]
        df_sc = df[[x, y]].dropna()
        if not df_sc.empty:
            fig_sc = px.scatter(df_sc, x=x, y=y, trendline="lowess", title=f"{y} vs {x}")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot.")

# Correlation heatmap among selected parameters
st.subheader("Correlation (Pearson)")
num_cols = [p for p in params if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Select at least two numeric parameters to view correlations.")

# --------------------------
# Alerts & Anomalies
# --------------------------

st.subheader("Alerts & Anomalies")
alerts = []
for key, rule in DEFAULT_THRESHOLDS.items():
    # find matching column in params
    matches = [p for p in params if p.lower() == key]
    if not matches:
        continue
    pcol = matches[0]
    if pcol in df.columns:
        vals = df[pcol].dropna()
        if vals.empty:
            continue
        if "min" in rule and (vals < rule["min"]).any():
            alerts.append(f"{pcol}: values below {rule['min']}")
        if "max" in rule and (vals > rule["max"]).any():
            alerts.append(f"{pcol}: values above {rule['max']}")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("No threshold alerts triggered for selected parameters.")

# Anomaly flagging per parameter
anom_tabs = st.tabs([f"Anomalies: {p}" for p in params])
for tab, p in zip(anom_tabs, params):
    with tab:
        if p not in df.columns or not pd.api.types.is_numeric_dtype(df[p]):
            st.info("Parameter not numeric.")
            continue
        dfa = df[[time_col, station_col, p]].dropna().sort_values(time_col)
        if dfa.empty:
            st.info("No data to analyze.")
            continue
        dfa["is_anom"] = zscore_anomalies(dfa[p], threshold=z_thr)
        n_anom = int(dfa["is_anom"].sum())
        st.write(f"Detected **{n_anom}** anomalies (z-score > {z_thr}).")
        if n_anom > 0:
            fig_a = px.scatter(dfa, x=time_col, y=p, color="is_anom", symbol="is_anom", title=f"{p} anomalies over time")
            st.plotly_chart(fig_a, use_container_width=True)
            st.dataframe(dfa.loc[dfa["is_anom"], :].reset_index(drop=True))
        else:
            st.info("No anomalies detected.")

# --------------------------
# Data Download
# --------------------------

st.subheader("Download Filtered Data")
out = df[[time_col, station_col, lat_col, lon_col] + [c for c in params if c in df.columns]].copy()
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name="filtered_water_quality.csv",
    mime="text/csv",
)

st.caption("Built with ❤️ in Streamlit. Replace demo with your CSV to explore your own data.")
