# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import os
# STRICT DARK THEME FOR ALL PLOTLY CHARTS
# STRICT DARK THEME FOR ALL PLOTLY CHARTS (BOLD AXES)
def apply_strict_dark_theme(fig, bg_color="#000000", font_color="white"):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(
            color=font_color,
            size=14
        ),
        title=dict(
            font=dict(color=font_color, size=18, family="Arial Black")
        ),
        legend=dict(
            font=dict(color=font_color, size=12)
        )
    )

    # X axis styling
    fig.update_xaxes(
        title_font=dict(color="white", size=14, family="Arial Black"),
        tickfont=dict(color="white", size=12, family="Arial Black"),
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.2)"
    )

    # Y axis styling
    fig.update_yaxes(
        title_font=dict(color="white", size=14, family="Arial Black"),
        tickfont=dict(color="white", size=12, family="Arial Black"),
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.2)"
    )

    return fig


st.set_page_config(layout="wide", page_title="Shipping Schedule — Charts")

# Minimal dark CSS for nicer contrast
st.markdown(
    """
    <style>
    /* App & sidebar background */
    .stApp {
        background-color: #0b0b0b;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b0b0b;
    }

    /* ===== SIDEBAR LABEL COLORS ===== */
    section[data-testid="stSidebar"] label {
        color: #8ecae6 !important;
        font-weight: 600;
    }

    /* Date input text */
    section[data-testid="stSidebar"] input {
        color: #ffffff !important;
        background-color: #000000 !important;
    }

    /* Multiselect selected values */
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* Dropdown menu background */
    div[role="listbox"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #000000;
        color: #8ecae6;
        border: 1px solid #8ecae6;
    }

    .stDownloadButton > button {
        background-color: #8ecae6;
        color: #000000;
        font-weight: bold;
    }

    /* Sidebar headers like "Data", "Filters" */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #8ecae6 !important;
    font-weight: 700;
}
/* ===== KPI METRIC COLOR & WEIGHT ===== */
div[data-testid="metric-container"] {
    background-color: #000000;
    border-radius: 8px;
    padding: 12px;
}

/* Metric label */
div[data-testid="metric-container"] label {
    color: #0096c7 !important;
    font-weight: 600;
}

/* Metric value */
div[data-testid="metric-container"] div {
    color: #0096c7 !important;
    font-weight: 800 !important;
    font-size: 26px !important;
}

/* ===== MAIN HEADERS ===== */
h1, h2 {
    color: #0096c7 !important;
    font-weight: 800 !important;
}

/* ===== SUBHEADERS ===== */
h3 {
    color: #ffdab9 !important;
    font-weight: 700 !important;
}


    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Matson Schedule Data Analysis Report")

# ------------------------
# Helpers
# ------------------------
def ensure_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique column names (append suffixes for duplicates)."""
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df

def safe_dt_parse(df: pd.DataFrame, col: str) -> pd.Series:
    """Return parsed datetime series for df[col] if present, else NaT series."""
    if col in df.columns:
        return pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    return pd.Series([pd.NaT] * len(df), index=df.index)

def build_counts_series_as_df(series, name='count'):
    """Given a datetime series or categorical series, return DataFrame with 'key' and 'count' columns."""
    vc = series.value_counts()
    out = vc.rename_axis('key').reset_index(name=name)
    return out

# ------------------------
# Load data (upload or local)
# ------------------------


st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader(
    "Upload shipping_schedule_enriched.csv (or .parquet)",
    type=["csv", "parquet"],
    accept_multiple_files=False
)

df = None

if uploaded is not None:
    try:
        if uploaded.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")
        st.stop()

# ⛔ STOP execution if df not loaded
if df is None:
    st.stop()


# ✅ SAFE to touch df from here onward
df = ensure_unique_cols(df)
df.columns = [c.strip() for c in df.columns]


# Defensive: unique column names & trim whitespace
df = ensure_unique_cols(df)
df.columns = [c.strip() for c in df.columns]

# ------------------------
# Clean / derive canonical fields
# ------------------------
# parse datetimes
df['arrive_dt'] = safe_dt_parse(df, 'arrive_dt')

# prefer final_imputed_depart_dt > depart_dt > depart
if 'final_imputed_depart_dt' in df.columns:
    df['final_depart'] = safe_dt_parse(df, 'final_imputed_depart_dt')
elif 'depart_dt' in df.columns:
    df['final_depart'] = safe_dt_parse(df, 'depart_dt')
elif 'depart' in df.columns:
    df['final_depart'] = safe_dt_parse(df, 'depart')
else:
    df['final_depart'] = pd.Series([pd.NaT]*len(df), index=df.index)

# Vessel name fallback
if 'Vessel_Name' not in df.columns and 'Vessel' in df.columns:
    df['Vessel_Name'] = df['Vessel'].astype(str)
elif 'Vessel_Name' not in df.columns:
    df['Vessel_Name'] = ''

# route field
if 'route' not in df.columns:
    df['route'] = df.get('OriginPortCode','').astype(str) + '-' + df.get('DestPortCode','').astype(str)

# vessvoy field
if 'vessvoy' not in df.columns:
    df['Voyage'] = df.get('Voyage','').astype(str)
    df['Bound'] = df.get('Bound','').astype(str)
    df['vessvoy'] = df['Vessel_Name'].astype(str).str.strip() + '*' + df['Voyage'].astype(str).str.strip() + '*' + df['Bound'].astype(str).str.strip()
else:
    df['vessvoy'] = df['vessvoy'].astype(str)

# port_call_index fallback
if 'port_call_index' not in df.columns:
    df = df.sort_values(['vessvoy','arrive_dt']).reset_index(drop=True)
    df['port_call_index'] = df.groupby('vessvoy').cumcount() + 1

# clean is_depart_suspicious -> boolean
if 'is_depart_suspicious' in df.columns:
    df['is_depart_suspicious'] = (
        df['is_depart_suspicious'].astype(str).str.strip().str.lower()
        .map({'true': True, 'false': False}).fillna(False).astype(bool)
    )
else:
    df['is_depart_suspicious'] = False

# transit_hours_final
try:
    df['transit_hours_final'] = (df['arrive_dt'] - df['final_depart']).dt.total_seconds() / 3600.0
except Exception:
    df['transit_hours_final'] = np.nan

# ports_in_voyage fallback
if 'ports_in_voyage' not in df.columns:
    df['ports_in_voyage'] = df.groupby('vessvoy')['route'].transform('nunique')

# create/normalize impute_method (defensive)
# prefer existing names; fallback to suggestions/auto detection
if 'impute_method' in df.columns:
    df['impute_method'] = df['impute_method'].fillna('').astype(str).str.strip()
else:
    df['impute_method'] = ''

# try suggestion columns
suggest_col = None
for c in ['suggestion_method','suggested_method','suggest_prev_method','suggested_imputed_method']:
    if c in df.columns:
        suggest_col = c
        break
if suggest_col:
    mask = (df['impute_method'] == '') & (df[suggest_col].notna())
    df.loc[mask, 'impute_method'] = df.loc[mask, suggest_col].astype(str).str.strip()

# ops-approved label
if 'ops_approved' in df.columns and 'suggested_imputed_depart_dt' in df.columns:
    ops_bool = df['ops_approved'].astype(str).str.lower().map({'true':True,'false':False}).fillna(False)
    mask_ops = (ops_bool == True) & (df['impute_method'] == '') & (df['suggested_imputed_depart_dt'].notna())
    df.loc[mask_ops, 'impute_method'] = 'ops_approved'

# mark auto_imputed where final differs from depart
if 'final_imputed_depart_dt' in df.columns and 'depart_dt' in df.columns:
    mask_auto = (df['impute_method'] == '') & (pd.notna(df['final_imputed_depart_dt'])) & (df['final_imputed_depart_dt'] != df['depart_dt'])
    df.loc[mask_auto, 'impute_method'] = 'auto_imputed'
# final fallback
df['impute_method'] = df['impute_method'].fillna('').replace('', 'none').astype(str)

# ------------------------
# Filters
# ------------------------
st.sidebar.header("Filters")
min_date = df['arrive_dt'].min()
max_date = df['arrive_dt'].max()
min_date_val = min_date.date() if pd.notna(min_date) else None
max_date_val = max_date.date() if pd.notna(max_date) else None
date_range = st.sidebar.date_input("Arrival date range", [min_date_val, max_date_val])

vessel_options = sorted(df['Vessel_Name'].dropna().unique().tolist()) if 'Vessel_Name' in df.columns else []
vessel_filter = st.sidebar.multiselect("Vessel_Name", options=vessel_options, default=None)
origin_options = sorted(df.get('OriginPortCode', pd.Series()).dropna().unique().tolist())
origin_filter = st.sidebar.multiselect("OriginPortCode", options=origin_options, default=None)
dest_options = sorted(df.get('DestPortCode', pd.Series()).dropna().unique().tolist())
dest_filter = st.sidebar.multiselect("DestPortCode", options=dest_options, default=None)


# apply filters
df_f = df.copy()
if date_range and len(date_range) == 2:
    start, end = date_range
    if pd.notna(start):
        df_f = df_f[df_f['arrive_dt'] >= pd.to_datetime(start)]
    if pd.notna(end):
        df_f = df_f[df_f['arrive_dt'] <= pd.to_datetime(end) + pd.Timedelta(days=1)]
if vessel_filter:
    df_f = df_f[df_f['Vessel_Name'].isin(vessel_filter)]
if origin_filter:
    df_f = df_f[df_f['OriginPortCode'].isin(origin_filter)]
if dest_filter:
    df_f = df_f[df_f['DestPortCode'].isin(dest_filter)]


# ------------------------
# Charts (render immediately)
# ------------------------
st.header("Overview & Charts")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows Count", f"{len(df_f):,}")
k2.metric("Unique voyages", f"{df_f['vessvoy'].nunique():,}")
#k3.metric("Suspicious rows", f"{df_f['is_depart_suspicious'].sum():,}")

k3.metric(
    "Unique Vessel Name",
    f"{df_f['Vessel_Name'].nunique():,}")
mean_transit = df_f['transit_hours_final'].dropna().mean()
k4.metric("Avg transit (hrs)", f"{mean_transit:.1f}" if not np.isnan(mean_transit) else "N/A")

# 1) Depart date distribution (line)
st.subheader("Depart Date Distribution (final depart)")
if df_f['final_depart'].notna().any():
    s = df_f['final_depart'].dropna().dt.floor('D')
    vc = s.value_counts()
    dep_counts = vc.rename_axis('date').reset_index(name='count')
    dep_counts['date'] = pd.to_datetime(dep_counts['date'], errors='coerce')
    dep_counts = dep_counts.sort_values('date')
    fig_depart = px.line(dep_counts, x='date', y='count', markers=True, title='Depart Count by Day')

    # enforce strict dark theme
    fig_depart = apply_strict_dark_theme(fig_depart, bg_color="#000000", font_color="white")

    st.plotly_chart(fig_depart, use_container_width=True)
else:
    st.info("No final depart datetimes available to plot.")

# ---- Insight: Depart Date Distribution ----
with st.container():
    counts = dep_counts['count']
    max_count = counts.max()
    mean_count = counts.mean()
    std_count = counts.std()

    if max_count > mean_count + 3 * std_count:
        st.warning(
            "⚠️ **Abnormal pattern detected**: "
            "One or more departure dates have unusually high counts, "
            "which may indicate bulk-loaded schedules or repeated depart dates."
        )
    elif std_count / mean_count > 0.8:
        st.info(
            "ℹ️ **High variability observed**: "
            "Departure activity varies significantly across dates."
        )
    else:
        st.success(
            "✅ **Normal distribution**: "
            "Departures are reasonably spread across dates with no extreme spikes."
        )

# 2) Arrivals over time (monthly)
st.subheader("Arrivals Over Time (monthly)")
if df_f['arrive_dt'].notna().any():
    arrivals = df_f.dropna(subset=['arrive_dt']).set_index('arrive_dt').resample('M').size().rename('count').reset_index()
    fig_arr = px.line(arrivals, x='arrive_dt', y='count', markers=True, title='Monthly Arrivals')
    fig_arr.update_layout(template='plotly_dark')
    st.plotly_chart(fig_arr, use_container_width=True)
else:
    st.info("No arrive_dt values to plot.")
    # ---- Insight: Arrival Trend ----
with st.container():
    month_counts = arrivals['count']
    pct_change = month_counts.pct_change().abs().max()

    if pct_change > 1.0:
        st.warning(
            "⚠️ **Sudden change detected**: "
            "Arrival volume shows sharp month-over-month variation."
        )
    elif month_counts.std() / month_counts.mean() > 0.6:
        st.info(
            "ℹ️ **Moderate variability**: "
            "Arrival volumes fluctuate but follow a general trend."
        )
    else:
        st.success(
            "✅ **Stable trend**: "
            "Arrivals show a consistent and predictable pattern over time."
        )


# 3) Top Origin & Destination bars
st.subheader("Top Origin & Destination Ports")
col1, col2 = st.columns(2)
with col1:
    if 'OriginPortCode' in df_f.columns and df_f['OriginPortCode'].notna().any():
        top_o = df_f['OriginPortCode'].value_counts().rename_axis('Origin').reset_index(name='count').head(20)
        fig_o = px.bar(top_o, x='count', y='Origin', orientation='h', title='Top Origin Ports', color='Origin')
        fig_o.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_o, use_container_width=True)
    else:
        st.info("OriginPortCode missing or empty.")
with col2:
    if 'DestPortCode' in df_f.columns and df_f['DestPortCode'].notna().any():
        top_d = df_f['DestPortCode'].value_counts().rename_axis('Dest').reset_index(name='count').head(20)
        fig_d = px.bar(top_d, x='count', y='Dest', orientation='h', title='Top Destination Ports', color='Dest')
        fig_d.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("DestPortCode missing or empty.")
        # ---- Insight: Port Concentration ----
total = df_f.shape[0]
top_origin_share = top_o['count'].iloc[0] / total if not top_o.empty else 0

if top_origin_share > 0.6:
    st.warning(
        "⚠️ **High concentration risk**: "
        "A single origin port accounts for more than 60% of movements."
    )
elif top_origin_share > 0.35:
    st.info(
        "ℹ️ **Moderate concentration**: "
        "Few ports dominate the network."
    )
else:
    st.success(
        "✅ **Well-distributed network**: "
        "Traffic is spread across multiple origin ports."
    )


# 4) Route matrix + heatmap
st.subheader("Route Frequency Matrix (pivot + heatmap)")
if ('OriginPortCode' in df_f.columns) and ('DestPortCode' in df_f.columns):
    pivot = pd.pivot_table(df_f, index='OriginPortCode', columns='DestPortCode', values='vessvoy', aggfunc='count', fill_value=0)
    st.dataframe(pivot)
    # heatmap if not huge
    if pivot.shape[0] <= 80 and pivot.shape[1] <= 80:
        try:
            hm = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, labels=dict(x='Dest', y='Origin', color='count'), aspect='auto', title='Route Heatmap (Origin x Dest)')
            hm.update_layout(template='plotly_dark')
            st.plotly_chart(hm, use_container_width=True)
        except Exception:
            pass
else:
    st.info("OriginPortCode / DestPortCode missing for route matrix.")
    

# 5) Port calls per vessel/voyage
st.subheader("Port Calls per Vessel/Voyage (top 50)")
calls = df_f.groupby('vessvoy').size().reset_index(name='calls').sort_values('calls', ascending=False).head(50)
if not calls.empty:
    fig_calls = px.bar(calls, x='calls', y='vessvoy', orientation='h', title='Port Calls per Vessel/Voyage', color='calls')
    fig_calls.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_calls, use_container_width=True)
else:
    st.info("No voyages to show.")

# 6) Transit time distribution (box) by top routes
st.subheader("Transit Time Distribution by Route (box)")
if df_f['transit_hours_final'].notna().any():
    top_routes = df_f['route'].value_counts().head(20).index.tolist()
    df_box = df_f[df_f['route'].isin(top_routes) & df_f['transit_hours_final'].notna()]
    if not df_box.empty:
        fig_box = px.box(df_box, x='route', y='transit_hours_final', title='Transit Hours by Top Routes')
        fig_box.update_layout(template='plotly_dark')
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Not enough transit data for box plot.")
else:
    st.info("No transit_hours_final present.")
# ---- Insight: Transit Time Variability ----
transit = df_box['transit_hours_final']

iqr = transit.quantile(0.75) - transit.quantile(0.25)
range_ratio = iqr / transit.median()

if range_ratio > 1.0:
    st.warning(
        "⚠️ **High transit variability**: "
        "Significant inconsistency observed across routes."
    )
elif range_ratio > 0.5:
    st.info(
        "ℹ️ **Moderate variability**: "
        "Some routes show inconsistent transit times."
    )
else:
    st.success(
        "✅ **Consistent transit performance**: "
        "Transit times are stable across major routes."
    )

# 7) Histogram: ports_in_voyage
st.subheader("Distribution: Ports per Voyage")
if 'ports_in_voyage' in df_f.columns and df_f['ports_in_voyage'].notna().any():
    fig_hist = px.histogram(df_f, x='ports_in_voyage', nbins=20, title='Ports per Voyage Distribution')
    fig_hist.update_layout(template='plotly_dark')
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("ports_in_voyage missing or empty.")
    # ---- Insight: Voyage Complexity ----
ports = df_f['ports_in_voyage']

if ports.max() > ports.mean() + 3 * ports.std():
    st.warning(
        "⚠️ **Highly complex voyages detected**: "
        "Some voyages involve unusually high number of port calls."
    )
elif ports.std() / ports.mean() > 0.7:
    st.info(
        "ℹ️ **Mixed service patterns**: "
        "Combination of direct and multi-stop voyages observed."
    )
else:
    st.success(
        "✅ **Standard voyage structure**: "
        "Most voyages follow consistent port-call patterns."
    )


# 8) Imputation methods pie (defensive)
#st.subheader("Imputation Methods (pie)")
#if 'impute_method' in df_f.columns and df_f['impute_method'].notna().any():
   # pie_df = df_f['impute_method'].fillna('none').astype(str).value_counts().reset_index(name='count').rename(columns={'index':'method'})
   # if not pie_df.empty:
        #fig_pie = px.pie(pie_df, names='impute_method', values='count', title='Imputation Methods')
       # fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        #fig_pie.update_layout(template='plotly_dark')
        #st.plotly_chart(fig_pie, use_container_width=True)
    #else:
       # st.info("No imputation method values to display.")
#else:
    #st.info("impute_method not present or empty — showing fallback 'none'")
    #fallback = pd.DataFrame({'method': ['none'], 'count':[len(df_f)]})
    #fig_pie = px.pie(fallback, names='method', values='count', title='Imputation Methods (none)')
    #fig_pie.update_layout(template='plotly_dark')
   # st.plotly_chart(fig_pie, use_container_width=True)

# 9) Scatter: transit vs port_call_index
st.subheader("Transit Hours vs Port Call Index (scatter)")
if df_f['transit_hours_final'].notna().any() and df_f['port_call_index'].notna().any():
    sc = df_f[df_f['transit_hours_final'].notna() & df_f['port_call_index'].notna()]
    fig_sc = px.scatter(sc, x='port_call_index', y='transit_hours_final', color='route', hover_data=['vessvoy','OriginPortCode','DestPortCode'], title='Transit Hours vs Port Call Index')
    fig_sc.update_layout(template='plotly_dark')
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("Not enough data for scatter chart.")
    # ---- Insight: Transit vs Port Call Index ----
with st.container():
    # compute correlation safely
    corr = sc[['port_call_index','transit_hours_final']].corr().iloc[0,1]

    outlier_threshold = sc['transit_hours_final'].quantile(0.95)
    outlier_ratio = (sc['transit_hours_final'] > outlier_threshold).mean()

    if corr > 0.6:
        st.warning(
            "⚠️ **Delay accumulation detected**: "
            "Transit times tend to increase as voyages progress, "
            "indicating possible cascading delays across port calls."
        )
    elif corr < -0.6:
        st.info(
            "ℹ️ **Decreasing transit pattern**: "
            "Later port calls show reduced transit times, "
            "possibly due to route optimization or shorter legs."
        )
    elif outlier_ratio > 0.1:
        st.warning(
            "⚠️ **High number of outliers detected**: "
            "Several port calls have unusually high transit times."
        )
    else:
        st.success(
            "✅ **Stable transit behavior**: "
            "Transit times remain consistent across port call sequence."
        )


# 10) Gantt-like timeline for selected voyages
st.subheader("Voyage Timeline (Gantt)")
vv_options = sorted(df_f['vessvoy'].dropna().unique().tolist())[:200]
sel_vv = st.multiselect("Select voyages to show (max 10)", options=vv_options, default=vv_options[:5])
if sel_vv:
    gantt_df = df_f[df_f['vessvoy'].isin(sel_vv) & df_f['arrive_dt'].notna() & df_f['final_depart'].notna()]
    if not gantt_df.empty:
        gantt_df = gantt_df.sort_values(['vessvoy','final_depart'])
        fig_g = px.timeline(gantt_df, x_start='final_depart', x_end='arrive_dt', y='vessvoy', color='route', hover_data=['OriginPortCode','DestPortCode'])
        fig_g.update_layout(template='plotly_dark')
        fig_g.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("No complete depart/arrive pairs for selected voyages.")
else:
    st.info("Select voyages to view timeline.")
    # ---- Insight: Voyage Timeline (Gantt) ----
with st.container():
    durations = (gantt_df['arrive_dt'] - gantt_df['final_depart']).dt.total_seconds() / 3600.0
    long_ratio = (durations > durations.quantile(0.95)).mean()

    # check overlaps within same vessvoy
    overlaps = 0
    for vv, g in gantt_df.groupby('vessvoy'):
        g = g.sort_values('final_depart')
        overlaps += (g['final_depart'].shift(-1) < g['arrive_dt']).sum()

    if overlaps > 0:
        st.warning(
            f"⚠️ **Schedule overlap detected**: "
            f"{overlaps} overlapping sailing windows found within selected voyages."
        )
    elif long_ratio > 0.15:
        st.warning(
            "⚠️ **Unusually long voyages detected**: "
            "Some voyages have significantly longer sailing durations than typical."
        )
    else:
        st.success(
            "✅ **Voyage schedules appear consistent**: "
            "No major overlaps or unrealistic durations detected."
        )


# 11) Data preview + download
st.subheader("Data preview (first 200 rows)")
preview_cols = ['vessvoy','Vessel_Name','Voyage','Bound','OriginPortCode','DestPortCode','arrive_dt','final_depart','transit_hours_final','is_depart_suspicious','impute_method']
present = [c for c in preview_cols if c in df_f.columns]
st.dataframe(df_f[present].head(200))

csv = df_f.to_csv(index=False)
st.download_button("Download filtered CSV", data=csv, file_name="shipping_schedule_filtered.csv", mime="text/csv")

st.markdown("Done — charts generated from your uploaded `shipping_schedule_enriched`.")
