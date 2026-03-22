# 📈 Stock Market Forecasting App

A Streamlit web app for stock trend prediction using the SARIMAX time-series model.

## Features
- Live stock data from Yahoo Finance (any global stock)
- Auto currency detection — ₹ for NSE, $ for NYSE, € for XETRA
- ADF stationarity test + seasonal decomposition
- SARIMA(2,1,2)(2,1,2,12) forecasting with interactive Plotly charts
- Hidden admin panel at `/?admin=1` for monitoring user searches

## How to Run
```bash
pip install -r requirements.txt
streamlit run Stock_Forecasting_final.py
```

## Tech Stack
Python · Streamlit · yfinance · statsmodels · Plotly · pandas

## Authors
**Poojan Patel** (12302110501080) · **Shrey Patel** (12302110501081)
Guided by Prof. Ritesh Upadhyay
G.H. Patel College of Engineering & Technology, CVM University
SEM 6 Mini Project · A.Y. 2025-26
