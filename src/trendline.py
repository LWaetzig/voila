from scipy.stats import linregress
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


# get data from api for 5 years back
data = yf.Ticker("AAPL").history("1mo")
data = data.drop(columns=["Volume" , "Dividends" , "Stock Splits"])

# data["numbers"] = data.index.map(dt.datetime.toordinal)
data["Number"] = np.arange(len(data)) + 1 

df = data.copy()

df_high_trend = df.copy()
df_low_trend = df.copy()

while len(df_high_trend) > 2:
    slope, intercept, r_value, p_value, std_err = linregress(x=df_high_trend["Number"], y=df_high_trend["High"])
    print(slope * df_high_trend["Number"] + intercept)
    df_high_trend = df_high_trend.loc[df_high_trend["High"] > slope * df_high_trend["Number"] + intercept]


while len(df_low_trend) > 2:
    slope, intercept , r_value ,p_value , std_err = linregress(x=df_low_trend["Number"] , y=df_low_trend["Low"])
    df_low_trend = df_low_trend.loc[df_low_trend["Low"] < slope * df_low_trend["Number"] + intercept]


slope, intercept, r_value, p_value, std_err = linregress(x=df_high_trend["Number"], y=df_high_trend["Close"])
df["Uptrend"] = slope * df["Number"] + intercept


slope, intercept, r_value, p_value, std_err = linregress(x=df_low_trend["Number"], y=df_low_trend["Close"])
df["Downtrend"] = slope * df["Number"] + intercept


fig , axis = plt.subplots(figsize=(20,15))

xdate = [x.date() for x in df.index]
axis.set_xlabel('Date')
axis.plot(xdate, df.Close, label="close", color="black")
axis.legend()

ax2 = axis.twiny() # ax2 and ax1 will have common y axis and different x axis, twiny
ax2.plot(df.Number, df.Uptrend, label="uptrend")
ax2.plot(df.Number, df.Downtrend, label="downtrend")

plt.legend()
plt.grid()
plt.show()