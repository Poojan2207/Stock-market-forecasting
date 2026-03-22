import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Stock AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================
# PREMIUM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    color: #00ffd5;
    text-align: center;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    box-shadow: 0 0 10px #00ffd5;
}

.stButton>button {
    background: linear-gradient(90deg,#00ffd5,#00ff88);
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


# =========================
# TITLE
# =========================

st.markdown("<h1>📈 AI Stock Forecast Dashboard</h1>", unsafe_allow_html=True)

st.write("Premium UI • SARIMAX Model • Streamlit")


# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Control Panel")

start_date = st.sidebar.date_input(
    "Start Date",
    date(2024,1,1)
)

end_date = st.sidebar.date_input(
    "End Date",
    date(2025,3,16)
)

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    "AAPL"
).upper()

if ticker == "":
    st.stop()


# =========================
# LOAD DATA
# =========================

data = yf.download(
    ticker,
    start=start_date,
    end=end_date
)

if data.empty:
    st.error("Wrong ticker")
    st.stop()


# ✅ FIX MULTIINDEX ERROR
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]


data["Date"] = data.index
data.reset_index(drop=True, inplace=True)


# =========================
# DATA TABLE
# =========================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Data")

st.dataframe(
    data,
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# GRAPH
# =========================

st.markdown('<div class="card">', unsafe_allow_html=True)

cols = [c for c in data.columns if c != "Date"]

fig = px.line(
    data,
    x="Date",
    y=cols,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# COLUMN SELECT
# =========================

column = st.selectbox(
    "Select column for forecast",
    cols
)

data = data[["Date", column]]


# =========================
# ADF TEST
# =========================

st.markdown('<div class="card">', unsafe_allow_html=True)

p = adfuller(data[column])[1]

if p < 0.05:
    st.success("Data is Stationary")
else:
    st.error("Data is NOT Stationary")

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# DECOMPOSITION
# =========================

st.markdown('<div class="card">', unsafe_allow_html=True)

decomp = seasonal_decompose(
    data[column],
    model="additive",
    period=12
)

st.plotly_chart(
    px.line(
        x=data["Date"],
        y=decomp.trend,
        title="Trend",
        template="plotly_dark"
    ),
    use_container_width=True
)

st.plotly_chart(
    px.line(
        x=data["Date"],
        y=decomp.seasonal,
        title="Seasonal",
        template="plotly_dark"
    ),
    use_container_width=True
)

st.plotly_chart(
    px.line(
        x=data["Date"],
        y=decomp.resid,
        title="Residual",
        template="plotly_dark"
    ),
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PARAMETERS
# =========================

st.subheader("⚙️ Model Settings")

c1,c2,c3 = st.columns(3)

with c1:
    p = st.slider("p",0,5,2)

with c2:
    d = st.slider("d",0,5,1)

with c3:
    q = st.slider("q",0,5,2)

season = st.slider(
    "Season",
    1,
    24,
    12
)


# =========================
# MODEL
# =========================

model = sm.tsa.statespace.SARIMAX(
    data[column],
    order=(p,d,q),
    seasonal_order=(p,d,q,season)
)

model = model.fit()


st.subheader("Model Summary")

st.text(model.summary())


# =========================
# FORECAST
# =========================

days = st.slider(
    "Forecast Days",
    1,
    365,
    10
)

pred = model.get_prediction(
    start=len(data),
    end=len(data)+days
)

pred = pred.predicted_mean

pred.index = pd.date_range(
    start=end_date,
    periods=len(pred),
    freq="D"
)

pred = pd.DataFrame(pred)
pred["Date"] = pred.index


# =========================
# FINAL GRAPH
# =========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data[column],
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=pred["Date"],
        y=pred["predicted_mean"],
        name="Forecast"
    )
)

fig.update_layout(
    template="plotly_dark",
    title="Actual vs Forecast"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown("""
### 👨‍💻 Project Team

Digant Kathiriya  
Aryan Kanada  
Dharmik Modhvadia
""")