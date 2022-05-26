import yfinance as yf
import plotly.graph_objs as go

def get_stock(stock_name = "DAX" , period="3y"):

    data = yf.Ticker(stock_name).history(period=period)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="stock data",
        )
    )

    stocks = dict()
    stocks["DAX"] = "DAX PERFORMANCE-INDEX"
    stocks["AAPL"] = "Apple Inc."
    stocks["TSLA"] = "Tesla Inc."
    stocks["AMZN"] = "Amazon.com, Inc."
    stocks["TWTR"] = "Twitter Inc"

    fig.update_layout(
        title=stocks[stock_name],
        yaxis_title = "Stock price (USD)"
        )

    fig.show()