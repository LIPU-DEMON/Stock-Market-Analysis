import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as psx
import seaborn as sns 
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly,plot_forecast_component_plotly,plot_weekly,plot_components
from plotly.subplots import make_subplots
st.header("Stock Analysis Demo",divider=True,width="stretch")
st.text("Predict ClosePrice on Various Stocks on different Models through Dashboard")
st.sidebar.text("Description: For Testing and Analysis Purpose only.")
st.sidebar.title("Select Model")
models = st.sidebar.selectbox("Select Model",["LSTM","Prophet","ARIMA","SARIMA"])
future = st.sidebar.slider("Choose Days to Predict",min_value=1,max_value=180)
col1,col2= st.columns(2)
with col1:
    stock = st.sidebar.radio("Choose Stock",["AAPL","MSFT","GE","IBM","JNJ"])
with col2:
    with st.sidebar.container(border=True,height=150):
        st.write("AAPL: Apple")
        st.write("MSFT: Microsoft")
        st.write("GE: GeneralElectric")
        st.write("IBM: IBM")
        st.write("JNJ: Jhonson&Johnson")
if models=="LSTM":
    if stock=="MSFT":
        model = joblib.load(f"Models/LSTM/MSFT.pkl")
        scaled = joblib.load(f"Models/scaled/{stock}_scaled.pkl")
        df = pd.read_csv("data/msft_scaled").iloc[:,2:11]
        past_values = df.tail(60).values.reshape(1,60,9)
        df = pd.read_csv("data/msft_scaled").set_index("Date")
    else:
        model = joblib.load(f"Models/LSTM/{stock}.pkl")
        scaled = joblib.load(f"Models/scaled/{stock}_scaled.pkl")
        df = pd.read_csv("data/Transformed.csv")
        past_values = df.tail(60).drop(columns=["Date"]).values.reshape(1,60,10)
        df = pd.read_csv("data/Transformed.csv").set_index("Date")

    prediction = model.predict(past_values)
    inverse = scaled.inverse_transform(prediction)
    st.write(f"Prediction for {future} days Close Price")
    forecast = inverse[0][:future]
    dates = pd.date_range(df.index[-1],periods=future)
    dataf = pd.DataFrame(forecast,columns=["Forecast"],index=dates)
    # st.write(dataf)
    st.subheader(f"Forecast")
    st.line_chart(dataf)
elif models=="ARIMA":
    model = joblib.load(f"Models/ARIMA/{stock}.pkl")
    df = pd.read_csv("data/EDA_transformed.csv").set_index("Date")["Close"]
    # past_values = df.tail(60).values
    prediction = list(model.forecast(steps=future))
    st.write(f"Prediction for {future} days Close Price")
    dates = pd.date_range(df.index[-1],periods=future)
    dataf = pd.DataFrame(prediction,columns=["Forecast"],index=dates)
    # st.write(dataf)
    st.subheader(f"Forecast")
    st.line_chart(dataf)

    #uncertainity levels (up and down)
    st.subheader("Reasonable Close Price")
    prediction = model.get_forecast(steps=future)
    forecast = model.forecast(future)
    df = prediction.conf_int()
    df["forecast"] = np.array(forecast)
    df.set_index(dates)
    # df.plot(xlabel="Index",ylabel="Close Intervals")
    st.line_chart(df)

elif models=="SARIMA":
    model = joblib.load(f"Models/SARIMA/{stock}.pkl")
    df = pd.read_csv("data/EDA_transformed.csv").set_index("Date")["Close"]
    prediction=list(model.forecast(steps=future))
    dates = pd.date_range(start="19-07-2025",periods=future)
    dataf = pd.DataFrame(prediction,columns=["forecast"],index=dates)
    st.subheader("Forecast")
    st.line_chart(dataf)

    #uncertainity levels (up and down)
    st.subheader("Reasonable Close Price")
    prediction = model.get_forecast(steps=future)
    forecast = model.forecast(future)
    reasonable = prediction.conf_int()
    df = pd.DataFrame(reasonable,columns=["lower_close","upper_close"],index=dates)
    df["forecast"] = np.array(forecast)
    # df.plot(xlabel="Index",ylabel="Close Intervals")
    st.line_chart(df)

elif models=="Prophet":
    model = joblib.load(f"Models/PROPHET/{stock}.pkl")
    dates = pd.date_range(start="19-07-2025",periods=future)
    prediction = model.make_future_dataframe(periods=future)
    forecast = model.predict(prediction)
    new_df = forecast.tail(future).iloc[:,[2,3,18]].set_index(dates)
    st.subheader("Overall Forecast")
    fig = plot_plotly(model,forecast)
    st.plotly_chart(fig)
    # st.pyplot(model.plot(forecast))
    #forecast

    fig1 = plot_forecast_component_plotly(model,forecast.iloc[(forecast.tail(1).index.start)-future:(forecast.tail(1).index.start)+future],name="yhat")
    fig1.update_layout(title="Forecast")
    st.plotly_chart(fig1)
    # st.line_chart(new_df["yhat"])
    # uncertainity levels
    st.subheader("Reasonable Close Price")
    st.line_chart(new_df)
    st.subheader("Components")
    fig2 = plot_components(model,forecast)
    st.pyplot(fig2)

    #Components




     