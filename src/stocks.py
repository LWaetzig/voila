import datetime as dt
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from Stock import Stock, Predictor

Stock = Stock("AAPL" , "5y")
df = Stock.get_stock_data()

Stock.visualize_data(df)

Predictor = Predictor(model="linreg" , deg=3)
predicted_df = Predictor.make_prediction(df)

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

    # set predicted_days depending on input
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

    # slice DataFrame to keep only columns "day" , "month" , "year"
    date_df = df[["day", "month", "year"]]

    # get latest date in DataFrame
    last_date = [list(date_df.iloc[-1])]
    future_dates = list()

    # append dataframe with future dates depending on input
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
    future_dates_df = pd.DataFrame(columns=["day", "month", "year"])

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
        future_dates_df.loc[i, "Date"] = dt.datetime.strptime(date_string, "%d/%m/%Y")

    future_dates_df = future_dates_df[["Date"]]
    # create ordinal numbers for datetime objects
    future_dates_df["date_value"] = future_dates_df["Date"].map(dt.datetime.toordinal)
    future_dates_df = future_dates_df.set_index(future_dates_df["Date"])
    future_dates_df = future_dates_df.drop(columns=["Date"])

    return future_dates_df


def evaluate_model(df, model = None , deg = None) -> None:
    """function to evaluate model using r2_score, mae, rmse

    Args:
        data (pd.DataFrame): original stock data
        deg (int): degree of regression
    """

    # only keep "Open" column in DataFrame
    df = df[["Close"]]

    # store ordinal values of datetime objects in X and stock values in y
    X = df.index.map(dt.datetime.toordinal)
    y = df["Close"]

    # split dataset into train and test batch
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if model == None:
        # create model using np regresseion of n-degree
        model = np.poly1d(np.polyfit(X_train, y_train, deg=deg))

    else:
        X_train = np.array(X_train).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)

        model.fit(X_train , y_train)
    
    # use model and predict values
    y_pred = model(X_test)

    # print out metrics for predicted data and evaluate model
    print("R^2 : ", r2_score(y_test, y_pred))
    print("MAE : ", mean_absolute_error(y_test, y_pred))
    print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))


# get data from api for 5 years back
data = yf.Ticker("AAPL").history("5y")

# clean dataframe from unnecessary columns
data = data.drop(columns=["Volume", "Dividends", "Stock Splits"])
data = data[["Close"]]
data["date_value"] = data.index.map(dt.datetime.toordinal)

# create prediciton model by using np regression
pred_model = np.poly1d(np.polyfit(data["date_value"], data["Close"], deg=3))

df = create_future_dates(data, months=1)

df["predicted"] = pred_model(df["date_value"])
df = df.drop(columns=["date_value"])

evaluate_model(data, model = "poly" , deg=3)

data = data.drop(columns=["date_value"])


lasso_model = Lasso(alpha=1)

evaluate_model(data , lasso_model)