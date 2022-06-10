import datetime as dt
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


class Predictor:
    def __init__(self, model, deg=None):
        self.model = model
        self.deg = deg if deg is not None else None

    def make_prediction(self, df) -> pd.DataFrame:
        """function to predict data using passed model

        Returns:
            pred_df (pd.DataFrame): pd.DataFrame with predicted values
        """

        if self.model == "linreg":
            pred_model = np.poly1d(
                np.polyfit(df["date_value"], df["Close"], deg=self.deg)
            )

            pred_df = create_future_dates(df, 31)

            pred_df["predicted"] = pred_model(pred_df["date_value"])
            pred_df = pred_df.drop(columns=["date_value"])

        return pred_df

    def trendline(self) -> pd.DataFrame:
        pass
        return pd.DataFrame

    def evaluate_model(self, df) -> None:
        """function to evaluate model using r2_score, mae, rmse

        Args:
            df (pd.DataFrame): dataset on which values were predicted
            model (_type_): prediction model used
        """

        df = df[["Close"]]

        # store ordinal values of datetime objects in X and stock values in y
        X = df.index.map(dt.datetime.toordinal)
        y = df["Close"]

        # split dataset into train and test batch
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if self.model == "linreg":
            # create regression model
            pred_model = np.poly1d(np.polyfit(X_train, y_train, deg=self.deg))

            # use model to prredict values
            y_pred = pred_model(X_test)

        # print out metrics for predicted data and evaluate model
        print("R^2 : ", r2_score(y_test, y_pred))
        print("MAE : ", mean_absolute_error(y_test, y_pred))
        print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))


def create_future_dates(df: pd.DataFrame, predict_interval: int) -> pd.DataFrame:
    """function to create future dates as datetime objects depending on passed days

    Args:
        df (pd.DataFrame): dataframe with future dates

    Returns:
        pd.DataFrame: dataframe with future dates
    """

    predicted_days = predict_interval

    # create columns for days, months, years from datetime index
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year

    df = df[["day", "month", "year"]]

    # drop column with old datetime objects
    df = df.reset_index(drop=True)

    # get latest date in DataFrame
    last_date = [list(df.iloc[-1])]
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
        series = pd.Series(liste, index=df.columns)
        future_dates_df = future_dates_df.append(series, ignore_index=True)

    # create datetime values from strings
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
