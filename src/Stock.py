import datetime as dt
import warnings
import plotly.graph_objs as go

import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


class Stock:
    def __init__(self, name, period):
        self.name = name
        self.period = period

    def get_stock_data(self) -> pd.DataFrame:
        """get stock data for a specific company downloaded using yahoo finance

        Args:
            name (str): international stock name (e.g. AAPL -> Apple Inc.)
            period (str): time interval for stock data (mo:= Months , y:=years , wk:=weeks, d:=days)

        Returns:
            data (pd.DataFrame): pd.DataFrame containing stock data from yahoo finance
        """

        if "d" in self.period:
            interval = "1m"
        else:
            interval = "1d"

        # download stock data from api
        data = yf.Ticker(self.name).history(self.period, interval)
        data = data[["Open", "Close", "High", "Low"]]

        # add mean value of stock price per day
        data["Mean"] = data.mean(axis="columns")

        if "d" not in self.period:
            data["mva"] = data["Open"].rolling(50).mean()

        # create numeric values for datetime objects
        data["date_value"] = data.index.map(dt.datetime.toordinal)

        return data

    def create_future_dates(self , df: pd.DataFrame) -> pd.DataFrame:
        """function to create future dates as datetime objects depending on number of passed days

        Args:
            df (pd.DataFrame): dataframe with future dates

        Returns:
            pd.DataFrame: dataframe with future dates
        """

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
        for i in range(8):
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
            future_dates_df.loc[i, "Date"] = dt.datetime.strptime(
                date_string, "%d/%m/%Y"
            )

        future_dates_df = future_dates_df[["Date"]]
        # create ordinal numbers for datetime objects
        future_dates_df["date_value"] = future_dates_df["Date"].map(
            dt.datetime.toordinal
        )
        future_dates_df = future_dates_df.set_index(future_dates_df["Date"])
        future_dates_df = future_dates_df.drop(columns=["Date"])

        # cut out first date which already has a value
        future_dates_df = future_dates_df.iloc[1: , :]

        return future_dates_df

    def make_prediction(self) -> pd.DataFrame:
        """function to predict data using passed model

        Returns:
            pred_df (pd.DataFrame): pd.DataFrame with predicted values
        """

        df = yf.Ticker(self.name).history("5y")
        df = df[["Open", "Close", "High", "Low"]]
        df["date_value"] = df.index.map(dt.datetime.toordinal)

        pred_model = np.poly1d(
            np.polyfit(df["date_value"], df["Close"], deg=3)
        )

        pred_df = self.create_future_dates(df)
        pred_df["predicted"] = pred_model(pred_df["date_value"])
        pred_df = pred_df.drop(columns=["date_value"])

        return pred_df


    def trendline(self) -> pd.DataFrame:
        pass


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

        # create regression model
        pred_model = np.poly1d(np.polyfit(X_train, y_train, deg=3))
        
        # use model to prredict values
        y_pred = pred_model(X_test)

        # print out metrics for predicted data and evaluate model
        print("R^2 : ", r2_score(y_test, y_pred))
        print("MAE : ", mean_absolute_error(y_test, y_pred))
        print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))


    def visualize_data(
        self, df: pd.DataFrame, moving_average=False, prediction=False
    ) -> None:
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

        if moving_average == True:
            fig.add_scatter(x=df.index, y=df["mva"], name="moving average", opacity=0.5)

        if prediction == True:
            fig.add_scatter(x=df.index, y=df["predicted"], name="trend")

        fig.add_scatter(x=df.index, y=df["Mean"], name="mean price", opacity=0.7)
        
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
            title=f"Stock for {stocks[self.name]}",
            yaxis_title="Stock price USD ($)",
            xaxis_title="Date",
        )
        
        fig.show()
