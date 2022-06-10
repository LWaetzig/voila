import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

from Prediction import Predictor

period = "7w"

if "d" in period:
    print(True)


def get_stock_data(stock_name : str , period : str) -> pd.DataFrame:
    """get stock data for a specific company downloaded using yahoo finance

    Args:
        name (str): international stock name (e.g. AAPL -> Apple Inc.)
        period (str): time interval for stock data (mo:= Months , y:=years , wk:=weeks, d:=days)

    Returns:
        data (pd.DataFrame): pd.DataFrame containing stock data from yahoo finance
    """

    if "d" in period:
        interval = "1m"
    else:
        interval = "1d"

    data = yf.Ticker(stock_name).history(period, interval)
    data = data[["Open", "Close", "High", "Low"]]

    # add mean value of stock price per day
    data["Mean"] = data.mean(axis="columns")

    if "d" not in period or "w" not in period:
        data["mva"] = data["Open"].rolling(50).mean()

    data["date_value"] = data.index.map(dt.datetime.toordinal)

    return data

df = get_stock_data("AAPL" , "1d")

def visualize_data(df: pd.DataFrame , stock_name : str, **add_args) -> None:
    """vizualisze stock data using plotly

    Args:
        df (pd.DataFrame): _description_
    """

    stocks = dict()
    stocks["ACN"] = "Accentur"
    stocks["SIE.DE"] = "Siemens AG"
    stocks["GME"] = "Gamestop"
    stocks["DOGE-USD"] = "Dogecoin :D"
    stocks["BTC-USD"] = "Bitcoin"
    stocks["DAX"] = "DAX PERFORMANCE-INDEX"
    stocks["AAPL"] = "Apple Inc."
    stocks["TSLA"] = "Tesla Inc."
    stocks["AMZN"] = "Amazon.com, Inc."
    stocks["TWTR"] = "Twitter Inc"

    fig = go.Figure()

    fig.add_scatter(x=df.index, y=df["Mean"], name="mean price", opacity=0.7)

    fig.add_scatter(
        x=df.index, y=df["mva"], name="moving average", opacity=0.5
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="stock price",
        )
    )

    fig.update_layout(
        title=f"Stock for {stocks[stock_name]}",
        yaxis_title="Stock price USD ($)",
        xaxis_title="Date",
    )

    fig.show()

visualize_data(df , "AAPL")


predictor = Predictor(model="linreg", deg=3)
predictor.evaluate_model(df)
predicted_df = predictor.make_prediction(df)