# Stock Market Forecasting App
# Made by Poojan Patel and Shrey Patel
# Mini Project - SEM 6

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import json
import os

st.set_page_config(
    page_title="Stock Forecast App",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------- admin stuff -------
# only poojan and shrey can login
ADMIN_USERS = {
    "poojan": "poojan123",
    "shrey":  "shrey123",
}

LOG_FILE = "search_log.json"

def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE) as f:
            return json.load(f)
    except:
        return []

def write_log(entry):
    logs = read_log()
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# map currency codes to symbols
CURRENCY_MAP = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£",
    "JPY": "¥", "CNY": "¥", "CAD": "CA$", "AUD": "A$",
    "HKD": "HK$", "SGD": "S$", "CHF": "CHF ", "KRW": "₩",
    "BRL": "R$", "MXN": "MX$",
}

def get_currency(ticker):
    try:
        info = yf.Ticker(ticker).info
        code = info.get("currency", "USD")
        return code, CURRENCY_MAP.get(code, code + " ")
    except:
        return "USD", "$"

# shared plotly styling so all charts look consistent
def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans, sans-serif", color="#4b5563", size=11),
        xaxis=dict(gridcolor="#f0f2f5", linecolor="#dde1e8", zeroline=False),
        yaxis=dict(gridcolor="#f0f2f5", linecolor="#dde1e8", zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#dde1e8", borderwidth=1),
        margin=dict(l=40, r=20, t=48, b=40),
    )
    return fig

# ------- styles -------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #f7f8fa;
    --white: #ffffff;
    --border: #dde1e8;
    --border2: #c8cdd7;
    --blue: #1a56db;
    --blue-light: #eff4ff;
    --green: #0e9f6e;
    --green-light: #f0fdf4;
    --red: #e02424;
    --amber: #b45309;
    --amber-light: #fffbeb;
    --text: #1e2532;
    --text2: #4b5563;
    --text3: #9ca3af;
    --serif: 'Playfair Display', Georgia, serif;
    --sans: 'IBM Plex Sans', sans-serif;
    --mono: 'IBM Plex Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}


/* inputs */
input, .stTextInput input, .stNumberInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.9rem !important;
}
input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,.1) !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: var(--white) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 8px !important;
}

/* buttons */
.stButton > button {
    background: var(--blue) !important;
    border: none !important;
    color: white !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.6rem !important;
    box-shadow: 0 2px 8px rgba(26,86,219,.2) !important;
    transition: all .15s ease !important;
}
.stButton > button:hover {
    background: #1648c0 !important;
    box-shadow: 0 4px 14px rgba(26,86,219,.3) !important;
    transform: translateY(-1px) !important;
}

/* metric cards */
[data-testid="metric-container"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.07) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.64rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text3) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.2rem !important;
    color: var(--text) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.07) !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.4rem 0 !important; }

/* section label like "HISTORICAL DATA ───────" */
.section {
    font-family: var(--sans);
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text3);
    margin: 2rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* navbar */
.navbar {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    margin-bottom: 1.4rem;
}
.app-title { font-family: var(--serif); font-size: 1.25rem; color: var(--text); }
.app-title span { color: var(--blue); }
.app-subtitle { font-family: var(--mono); font-size: 0.65rem; color: var(--text3); }

/* input card */
.input-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.8rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    margin-bottom: 1.6rem;
}
.input-card-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 1rem;
}
.field-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text2);
    margin-bottom: 3px;
    display: block;
}

/* admin login */
.admin-card {
    width: 100%;
    max-width: 380px;
    background: var(--white);
    border: 1px solid var(--border);
    border-top: 3px solid var(--amber);
    border-radius: 12px;
    padding: 2.5rem 2.2rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.1);
    margin: 10vh auto 0;
}
.admin-eyebrow {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.5rem;
}
.admin-title { font-family: var(--serif); font-size: 1.6rem; color: var(--text); margin-bottom: 0.3rem; }
.admin-sub { font-size: 0.76rem; color: var(--text3); margin-bottom: 1.8rem; }

/* admin dashboard header */
.admin-header {
    background: #1e2532;
    border-radius: 10px;
    padding: 0.85rem 1.6rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.admin-header-title { font-family: var(--serif); font-size: 1.1rem; color: #f7f8fa; }
.admin-header-title span { color: #fcd34d; }
.admin-tag {
    background: var(--amber-light);
    color: var(--amber);
    border: 1px solid #fcd34d;
    border-radius: 20px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.22rem 0.8rem;
}

.stationary-yes { color: #0e9f6e; font-family: var(--mono); font-weight: 600; }
.stationary-no  { color: #e02424; font-family: var(--mono); font-weight: 600; }

.forecast-title {
    font-family: var(--serif);
    font-size: 1.4rem;
    color: var(--text);
    border-bottom: 2px solid var(--blue);
    padding-bottom: 0.35rem;
    margin: 1.8rem 0 1rem;
    display: inline-block;
}

.footer {
    text-align: center;
    font-size: 0.74rem;
    color: var(--text3);
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer strong { color: var(--text2); }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



params = st.query_params
on_admin_page = params.get("admin", "") == "1"


# =============================================
# ADMIN - Login page
# =============================================
def admin_login():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("""
        <div class="admin-card">
            <div class="admin-eyebrow">Backend Access</div>
            <div class="admin-title">Admin Panel</div>
            <div class="admin-sub">Only for Admins</div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="field-label">Username</span>', unsafe_allow_html=True)
        uname = st.text_input("u", placeholder="enter your name",
                               label_visibility="collapsed", key="au")
        st.markdown('<span class="field-label" style="margin-top:0.6rem">Password</span>',
                    unsafe_allow_html=True)
        pwd = st.text_input("p", placeholder="password",
                             type="password", label_visibility="collapsed", key="ap")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login →", use_container_width=True):
            u = uname.strip().lower()
            if u in ADMIN_USERS and ADMIN_USERS[u] == pwd:
                st.session_state["admin_in"] = True
                st.session_state["admin_name"] = u
                st.rerun()
            else:
                st.error("Wrong credentials.")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================
# ADMIN - Dashboard
# =============================================
def admin_dashboard():
    name = st.session_state.get("admin_name", "admin")

    st.markdown(f"""
    <div class="admin-header">
        <div class="admin-header-title">Stock Forecast <span>— Admin</span></div>
        <div style="display:flex;align-items:center;gap:0.9rem;">
            <span class="admin-tag">⭑ Admin</span>
            <span style="font-family:var(--mono);font-size:0.72rem;color:#9ca3af">{name}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    logs = read_log()

    # quick stats at the top
    total = len(logs)
    unique_tickers = len(set(e["ticker"] for e in logs))
    last_seen = logs[-1]["timestamp"] if logs else "nothing yet"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Searches", total)
    c2.metric("Unique Tickers", unique_tickers)
    c3.metric("Last Activity", last_seen)

    st.markdown('<div class="section">Search History</div>', unsafe_allow_html=True)

    if not logs:
        st.info("No one has searched anything yet. Come back later.")
    else:
        for entry in reversed(logs):
            label = f"{entry.get('ticker','?')}  ·  {entry.get('timestamp','')}"
            with st.expander(label, expanded=False):
                a, b, c = st.columns(3)
                a.markdown(f"**Date Range**<br>{entry.get('start_date','')} → {entry.get('end_date','')}", unsafe_allow_html=True)
                b.markdown(f"**Column**<br>{entry.get('column','')}", unsafe_allow_html=True)
                c.markdown(f"**Forecast Days**<br>{entry.get('forecast_days','')}", unsafe_allow_html=True)

                p = entry.get('arima_p','')
                d = entry.get('arima_d','')
                q = entry.get('arima_q','')
                s = entry.get('seasonal','')

                st.markdown(f"""
                <div style="margin:0.7rem 0 0.3rem;font-size:0.7rem;font-weight:700;
                            color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em">ARIMA params</div>
                <code style="background:#f0f2f5;padding:0.2rem 0.5rem;border-radius:4px;margin-right:4px">p={p}</code>
                <code style="background:#f0f2f5;padding:0.2rem 0.5rem;border-radius:4px;margin-right:4px">d={d}</code>
                <code style="background:#f0f2f5;padding:0.2rem 0.5rem;border-radius:4px;margin-right:4px">q={q}</code>
                <code style="background:#f0f2f5;padding:0.2rem 0.5rem;border-radius:4px">seasonal={s}</code>
                """, unsafe_allow_html=True)

                if entry.get("model_summary"):
                    st.markdown('<div style="margin-top:0.9rem;font-size:0.7rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em">Model Summary</div>', unsafe_allow_html=True)
                    st.code(entry["model_summary"], language="text")

        # let them download the log
        st.markdown('<div class="section">Export</div>', unsafe_allow_html=True)
        df = pd.DataFrame(logs).drop(columns=["model_summary"], errors="ignore")
        st.download_button(
            "Download Log as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "search_log.csv",
            "text/csv"
        )

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.pop("admin_in", None)
        st.session_state.pop("admin_name", None)
        st.rerun()


# =============================================
# PUBLIC APP
# =============================================
def main_app():
    # navbar
    st.markdown("""
    <div class="navbar">
        <div>
            <div class="app-title">Stock<span>Cast</span></div>
            <div class="app-subtitle">Market Forecasting · Powered by SARIMAX · Data from Yahoo Finance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # input fields
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-label">📊 Enter Details</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1.0])
    with c1:
        st.markdown('<span class="field-label">Start Date</span>', unsafe_allow_html=True)
        start_date = st.date_input("sd", value=None, min_value=date(2000, 1, 1),
                                    max_value=date.today(), label_visibility="collapsed", key="sd")
    with c2:
        st.markdown('<span class="field-label">End Date</span>', unsafe_allow_html=True)
        end_date = st.date_input("ed", value=None, min_value=date(2000, 1, 1),
                                  max_value=date.today(), label_visibility="collapsed", key="ed")
    with c3:
        st.markdown('<span class="field-label">Ticker Symbol</span>', unsafe_allow_html=True)
        ticker = st.text_input("tk", placeholder="e.g. AAPL, TSLA, RELIANCE.NS, TCS.NS",
                                label_visibility="collapsed", key="tk").strip().upper()
    with c4:
        st.markdown('<span class="field-label">Forecast Days</span>', unsafe_allow_html=True)
        forecast_days = st.number_input("fd", 1, 365, 10, label_visibility="collapsed", key="fd")

    st.markdown('</div>', unsafe_allow_html=True)

    # basic validation
    if not ticker or not start_date or not end_date:
        st.markdown("""
        <div style="background:#eff4ff;border:1px solid #bfdbfe;border-left:3px solid #1a56db;
                    border-radius:8px;padding:0.8rem 1.1rem;font-size:0.84rem;color:#4b5563;">
            👆 Fill in the <strong>start date</strong>, <strong>end date</strong> and
            <strong>ticker symbol</strong> to get started.
            Try <code>AAPL</code> or <code>RELIANCE.NS</code>
        </div>""", unsafe_allow_html=True)
        return

    if start_date >= end_date:
        st.error("Start date has to be before end date. Please fix that.")
        return

    # download data
    with st.spinner(f"Loading {ticker} data..."):
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error(f"Couldn't find data for **{ticker}**. Double check the symbol — Indian stocks need .NS (e.g. RELIANCE.NS)")
        return

    # flatten multi-level columns yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns]

    df.insert(0, "Date", df.index)
    df.reset_index(drop=True, inplace=True)

    # figure out what currency this stock trades in
    currency_code, currency_sym = get_currency(ticker)

    # summary metrics at the top
    close_col = next((c for c in df.columns if "Close" in c), None)
    if close_col:
        price_now = float(df[close_col].iloc[-1])
        price_then = float(df[close_col].iloc[0])
        change = price_now - price_then
        change_pct = (change / price_then) * 100
        highest = float(df[close_col].max())
        lowest = float(df[close_col].min())
        average = float(df[close_col].mean())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Price", f"{currency_sym}{price_now:,.2f} ({currency_code})")
        m2.metric("Total Change", f"{change_pct:+.2f}%", f"{currency_sym}{change:+,.2f}")
        m3.metric("Period High", f"{currency_sym}{highest:,.2f}")
        m4.metric("Period Low", f"{currency_sym}{lowest:,.2f}")
        m5.metric("Average", f"{currency_sym}{average:,.2f}")

    # show the raw data
    st.markdown('<div class="section">Historical Data</div>', unsafe_allow_html=True)
    st.caption(f"{len(df)} trading days between {start_date} and {end_date}")
    st.dataframe(df, use_container_width=True, height=210)

    # price chart for all columns
    st.markdown('<div class="section">Price Chart</div>', unsafe_allow_html=True)
    all_cols = [c for c in df.columns if c != "Date"]
    fig = px.line(df, x="Date", y=all_cols, title=f"{ticker} — All Columns", height=420)
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    # pick which column to use for forecasting
    st.markdown('<div class="section">Pick Column to Forecast</div>', unsafe_allow_html=True)
    chosen_col = st.selectbox("column", all_cols, label_visibility="collapsed")
    df = df[["Date", chosen_col]]

    # ADF stationarity test
    st.markdown('<div class="section">Stationarity Check (ADF Test)</div>', unsafe_allow_html=True)
    adf_pval = adfuller(df[chosen_col])[1]
    if adf_pval < 0.05:
        st.markdown('<p class="stationary-yes">✔ Data is stationary (p &lt; 0.05)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stationary-no">✘ Data is not stationary (p ≥ 0.05)</p>', unsafe_allow_html=True)

    # seasonal decomposition - needs at least 24 points
    st.markdown('<div class="section">Seasonal Decomposition</div>', unsafe_allow_html=True)

    MIN_POINTS = 24
    if len(df) < MIN_POINTS:
        st.markdown(f"""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-left:3px solid #b45309;
                    border-radius:8px;padding:0.85rem 1.1rem;font-size:0.84rem;color:#78350f;">
            ⚠️ <strong>Not enough data to decompose.</strong><br>
            This needs at least <strong>24 data points</strong> but you only have <strong>{len(df)}</strong>.
            Try selecting at least 2-3 months of data.
        </div>""", unsafe_allow_html=True)
    else:
        decomp = seasonal_decompose(df[chosen_col], model='additive', period=12)

        def small_chart(vals, title, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=vals, mode='lines',
                                     line=dict(color=color, width=1.8)))
            fig.update_layout(title=dict(text=title, font=dict(size=12, color="#1e2532")), height=240)
            style_chart(fig)
            return fig

        col1, col2, col3 = st.columns(3)
        with col1: st.plotly_chart(small_chart(decomp.trend, "Trend", "#1a56db"), use_container_width=True)
        with col2: st.plotly_chart(small_chart(decomp.seasonal, "Seasonal", "#0e9f6e"), use_container_width=True)
        with col3: st.plotly_chart(small_chart(decomp.resid, "Residuals", "#e02424"), use_container_width=True)

    # SARIMA model
    if len(df) < MIN_POINTS:
        st.markdown("""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-left:3px solid #b45309;
                    border-radius:8px;padding:0.85rem 1.1rem;font-size:0.84rem;color:#78350f;">
            ⚠️ <strong>Forecast needs more data.</strong>
            Select at least 2-3 months to run the model.
        </div>""", unsafe_allow_html=True)
        return

    p, d, q = 2, 1, 2
    seasonal = 12

    with st.spinner("Running SARIMA model, this might take a moment..."):
        try:
            sarima = sm.tsa.statespace.SARIMAX(
                df[chosen_col],
                order=(p, d, q),
                seasonal_order=(p, d, q, seasonal)
            ).fit(disp=False)
        except Exception:
            st.markdown("""
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-left:3px solid #e02424;
                        border-radius:8px;padding:0.85rem 1.1rem;font-size:0.84rem;color:#7f1d1d;">
                ⚠️ <strong>Model didn't converge.</strong>
                Try a longer date range — 3 to 6 months usually works well.
            </div>""", unsafe_allow_html=True)
            return

    # quietly log this search for admin
    try:
        write_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "column": chosen_col,
            "arima_p": p, "arima_d": d, "arima_q": q,
            "seasonal": seasonal,
            "forecast_days": int(forecast_days),
            "model_summary": str(sarima.summary()),
        })
    except:
        pass  # don't break the app if logging fails

    # forecast
    st.markdown('<div class="forecast-title">Forecast</div>', unsafe_allow_html=True)

    future = sarima.get_prediction(start=len(df), end=len(df) + forecast_days)
    future_vals = future.predicted_mean
    future_vals.index = pd.date_range(start=end_date, periods=len(future_vals), freq='D')

    future_df = pd.DataFrame({
        "Date": future_vals.index,
        "Predicted": future_vals.values
    }).reset_index(drop=True)

    # combined chart
    combined = go.Figure()
    combined.add_trace(go.Scatter(
        x=df["Date"], y=df[chosen_col],
        mode='lines', name='Historical',
        line=dict(color='#1a56db', width=1.8)
    ))
    combined.add_trace(go.Scatter(
        x=future_df["Date"], y=future_df["Predicted"],
        mode='lines', name='Forecast',
        line=dict(color='#0e9f6e', width=2, dash='dot'),
        fill='tozeroy', fillcolor='rgba(14,159,110,0.04)'
    ))
    combined.update_layout(
        title=f"{ticker} — Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_code})",
        height=460
    )
    style_chart(combined)
    st.plotly_chart(combined, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section">Forecast Values</div>', unsafe_allow_html=True)
        st.dataframe(future_df, use_container_width=True)
    with right:
        st.markdown('<div class="section">Recent Data (last 20 days)</div>', unsafe_allow_html=True)
        st.dataframe(df.tail(20), use_container_width=True)

    if st.button("Show Separate Charts"):
        h = px.line(df, x="Date", y=chosen_col, title="Historical", height=300)
        h.update_traces(line_color="#1a56db", line_width=1.8)
        style_chart(h)
        st.plotly_chart(h, use_container_width=True)

        p2 = px.line(future_df, x="Date", y="Predicted", title="Forecast Only", height=300)
        p2.update_traces(line_color="#0e9f6e", line_dash="dot", line_width=2)
        style_chart(p2)
        st.plotly_chart(p2, use_container_width=True)

    st.markdown("""
    <div class="footer">
        Made by <strong>Poojan Patel</strong> & <strong>Shrey Patel</strong> · SEM 6 Mini Project
    </div>
    """, unsafe_allow_html=True)


# route to the right page
if on_admin_page:
    if not st.session_state.get("admin_in"):
        admin_login()
    else:
        admin_dashboard()
else:
    main_app()