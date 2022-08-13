# STREAMLIT

# cd "/Users/hp/OneDrive/Documents/Python Anaconda/Streamlit_Stock_App"
# streamlit run AppPredictionStock.py

# pip install streamlit --upgrade 0.79

import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
# from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

def main():
    
    st.title("Stock Prediction")

    # Image
    from PIL import Image
    image = Image.open('./StockCorrected.jpg')
    st.image(image, caption='Stock Prediction with FB Prophet', use_column_width=True)

# LOADING
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ("FB", "AAPL", "AMZN", "MSFT", "GOOG", "HON", "DTE", "VOD", "HEI", "SAP")
    selected_stock = st.selectbox("Select STOCK", stocks)

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        data['MA'] = data.Close.rolling(30).mean()
        return data

    data = load_data(selected_stock)

    def stock_names(x):
        if x == "FB":
            return "Facebook"
        elif x == "AAPL":
            return "Apple"
        elif x == "AMZN":
            return "Amazon"
        elif x == "MSFT":
            return "Microsoft"
        elif x == "GOOG":
            return "Google"
        elif x == "HON":
            return "Honeywell"
        elif x == "DTE":
            return "Deutsche Telekom"
        elif x == "VOD":
            return "Vodafone"
        elif x == "HEI":
            return "Heidelberg Cement"
        elif x == "SAP":
            return "SAP"

    st.subheader(f"{stock_names(selected_stock)}")
    st.write(data.tail(5))

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price', line=dict(color="black")))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MA'], name='Monthly average', line=dict(color="#C125B5")))

        fig.layout.update(title_text=f"{stock_names(selected_stock)} Plot", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_raw_data()

# FORECAST
    style1 = '<p style="color:Red; font-size: 20px;">Loading data...</p>'
    style2 = '<p style="color:Green; font-size: 20px;">Loading data... DONE!</p>'
    data_load_state = st.markdown(style1, unsafe_allow_html=True)

    n_days = st.slider("Days of prediction:", 1, 1000, 180)
    period = n_days

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet(weekly_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    data_load_state.markdown(style2, unsafe_allow_html=True)
    
    st.subheader(f"{stock_names(selected_stock)} Forecast")
    st.write(forecast.tail(5))

    # st.write(f"{stock_names(selected_stock)} Forecast")
    # fig1 = plot_plotly(m, forecast)
    # st.plotly_chart(fig1)

    def plot_forecast_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price', line=dict(color="black")))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Close price (Forecast)', line=dict(color="#3395FF")))

        fig.layout.update(title_text=f"{stock_names(selected_stock)} Forecast Plot", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_forecast_data()

    # st.write(f"{stock_names(selected_stock)} Forecast Components")
    # fig2 = m.plot_components(forecast)
    # st.write(fig2)

    st.subheader("Sources:")
    st.write("Source of the data for the model: https://finance.yahoo.com")
    st.write("To see other author’s projects: https://jaroslavkotrba.com")

if __name__ == '__main__':
    main()
