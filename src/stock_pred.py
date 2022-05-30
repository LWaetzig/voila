import datetime as dt
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

warnings.simplefilter("ignore")

def create_future_dates(df, days=None, weeks=None, months=None) -> pd.DataFrame:
    """function to create future dates as datetime objects depending on passed days

    Args:
        df (pd.DataFrame): dataframe with datetime as index
        days (int , optional): number of days to be predicted. Default is None
        weeks (int , optional): number of weeks to be predicted. Default is None
        months (int , optional): number of months to be predicted. Default is None

    Returns:
        pd.DataFrame: dataframe with future dates
    """

    if days:
        predicted_days = days
    elif weeks:
        predicted_days = weeks * 7
    elif months:
        predicted_days = months * 31
    
    # create columns for days, months, years from datetime index
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df = df.reset_index()

    date_df = df[["day", "month", "year"]]

    last_date = [list(date_df.iloc[-1])]
    future_dates = list()

    # append dataframe with future dates
    for i in range(predicted_days + 1):
        appended = [(last_date[0][0] + i), last_date[0][1], last_date[0][2]]
        if appended[0] > 31 and appended[1] > 12:
            appended[0] = 1
            appended[1] = 1
            appended[2] = appended[2] + 1

        if (
            appended[1] == 1
            or appended[1] == 3
            or appended[1] == 5
            or appended[1] == 7
            or appended[1] == 8
            or appended[1] == 10
            or appended[1] == 12
        ):
            if appended[0] > 31:
                appended[0] = appended[0] - 31
                appended[1] = appended[1] + 1

        elif appended[1] == 2:
            if appended[0] > 28:
                appended[0] = appended[0] - 28
                appended[1] = appended[1] + 1

        else:
            if appended[0] > 30:
                appended[0] = appended[0] - 28
                appended[1] = appended[1] + 1

        future_dates.append(appended)

    # create new dataframe for future dates
    future_dates_df = pd.DataFrame(columns=["day" , "month" , "year"])
    # append new dataframe with new dates
    for liste in future_dates:
        series = pd.Series(liste, index=date_df.columns)
        future_dates_df = future_dates_df.append(series, ignore_index=True)

    # create datetime values
    for i in range(len(future_dates_df)):
       day = str(future_dates_df.loc[i, "day"])
       month = str(future_dates_df.loc[i, "month"])
       year = str(future_dates_df.loc[i, "year"])

       date_string = f"{day}/{month}/{year}"
       future_dates_df.loc[ i , "Date"] = dt.datetime.strptime(date_string, "%d/%m/%Y")

    future_dates_df = future_dates_df[["Date"]]
    future_dates_df["date_value"] = future_dates_df["Date"].map(dt.datetime.toordinal)
    future_dates_df = future_dates_df.set_index(future_dates_df["Date"])
    future_dates_df = future_dates_df.drop(columns=["Date"])

    return future_dates_df


# get data from api for 5 years back
data = yf.Ticker("AAPL").history("5y")
stock_data = data.copy()
# clean dataframe from unnecessary columns
data = data.drop(columns=["Volume", "Dividends", "Stock Splits"])
data = data[["Close"]]
data["date_value"] = data.index.map(dt.datetime.toordinal)

# create prediciton model by using np regression
pred_model = np.poly1d(np.polyfit(data["date_value"], data["Close"], deg=3))

df = create_future_dates(data , months=1)

df["predicted"] = pred_model(df["date_value"])
df = df.drop(columns=["date_value"])

r2 = r2_score()



stock_data = stock_data[["Open"]]

X = stock_data.index.map(dt.datetime.toordinal)
y = stock_data["Open"]

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.2)

model = np.poly1d( np.polyfit(X_train , y_train , deg = 1))
y_pred = model(X_test)

print( "R^2 : " , r2_score(y_test , y_pred))
print( "MAE : " , mean_absolute_error(y_test , y_pred))
print( "RMSE : " , np.sqrt(mean_squared_error(y_test , y_pred)))


sgd_model = SGDRegressor()
y_sgd_pred = model(X_test)

print( "R^2 : " , r2_score(y_test , y_sgd_pred))
print( "MAE : " , mean_absolute_error(y_test , y_sgd_pred))
print( "RMSE : " , np.sqrt(mean_squared_error(y_test , y_sgd_pred)))

data = data.drop(columns=["date_value"])

fig = go.Figure()
fig2 = go.Figure()
#fig = fig.add_trace(
#    go.Candlestick(
#        x=data.index,
#        close=data["Close"],
 #       name = "actual values"
#    )

#)
fig = fig.add_trace(
    go.Candlestick(
        x = stock_data.index,
        open = stock_data["Open"],
        high = stock_data["High"],
        low = stock_data["Low"],
        close = stock_data["Close"],
        name = "stock price"
    )
)

fig2 = fig2.add_scatter(
    x = df.index,
    y = df["predicted"],
    name = "predicted"
)

fig.show()
fig2.show()

