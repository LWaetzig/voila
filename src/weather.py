os.chdir("..")
os.getcwd()

import datetime as dt
import json
import os

import geopandas as gpd
import geopy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import requests
import numpy as np
import statistics
from IPython.display import display, Image

icons = {
    "Clear": "figures/sun.png",
    "Rain": "figures/rain.png",
    "sun_cloud": "figures/sun_cloud.png",
    "Clouds": "figures/cloud.png",
    "wind": "figures/wind.png",
}

# define global vairables
API_KEY = "0903f43c667c445d4a0c16920ef81c36"
SHAPE_FILE = "data/germany_geo.json"

# input for city
# city_name = str(input("give me a city name: "))


def get_weather_data_city(api_key: str, city_name: str) -> dict:
    """function to get raw weather data for specific city

    Args:
        city_name (str): name of city in germany
        api_key (str): necessary authentification to create api request

    Returns:
        dict: dictionary with data raw weather data from api
    """

    # get longitude and latitude for given city
    service = geopy.Nominatim(user_agent="myGeocoder")
    Location = service.geocode(f"{city_name}, Germany")

    # try to request api and store weather data
    try:
        url = (
            "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric"
            % (Location.latitude, Location.longitude, api_key)
        )
        response = requests.get(url)
        data = json.loads(response.text)
    except Exception as e:
        print(f"something went wrong: {e}")

    # if there is a failure code in the dataset stop and return None
    if "cod" in data.keys():
        print(data["cod"], data["message"])
        return None
    else:
        return data


def get_granularity(data: dict, granularity: str) -> pd.DataFrame:
    """choose hourly data from weather data set and store in DataFrame

    Args:
        city_name (str): name of city in germany
        api_key (str): necessary authentification to create api request
        granularity (str): hourly and daily

    Returns:
        pd.DataFrame: filled DataFrame with weather data from api
    """

    # only select hourly data from api request
    timezone = data["timezone"]
    forecast = data[granularity]

    # create, fill DataFrame for weather data and set timestamps as index
    weather_df = pd.DataFrame()

    if granularity == "hourly":

        weather_df["date"] = [
            dt.datetime.fromtimestamp(entry["dt"], pytz.timezone(timezone))
            for entry in forecast
        ]
        weather_df["temp"] = [entry["temp"] for entry in forecast]
        weather_df["feels_like"] = [entry["feels_like"] for entry in forecast]
        weather_df["clouds"] = [entry["clouds"] for entry in forecast]
        weather_df["weather"] = [entry["weather"][0]["main"] for entry in forecast]
        weather_df = weather_df.set_index("date", drop=True)

    elif granularity == "daily":
        weather_df["date"] = [
            dt.datetime.fromtimestamp(entry["dt"], pytz.timezone(timezone))
            for entry in forecast
        ]
        weather_df["day_temp"] = [entry["temp"]["day"] for entry in forecast]
        weather_df["night_temp"] = [entry["temp"]["night"] for entry in forecast]
        weather_df["min_temp"] = [entry["temp"]["min"] for entry in forecast]
        weather_df["max_temp"] = [entry["temp"]["max"] for entry in forecast]
        weather_df["feels_like"] = [
            round(statistics.mean(
                (entry["feels_like"]["day"],
                entry["feels_like"]["night"],
                entry["feels_like"]["eve"],
                entry["feels_like"]["morn"])
            ) , 2)
            for entry in forecast
        ]
        weather_df["wind_speed"] = [entry["wind_speed"] for entry in forecast]
        weather_df["pressure"] = [entry["pressure"] for entry in forecast]
        weather_df["uvi"] = [entry["uvi"] for entry in forecast]
        weather_df["weather"] = [entry["weather"][0]["main"] for entry in forecast]
        weather_df["clouds"] = [entry["clouds"] for entry in forecast]
        weather_df = weather_df.set_index("date" , drop=True)

    return weather_df

# get and prepare data
data = get_weather_data_city(API_KEY, city_name="Dresden")
df_daily = get_granularity(data , "daily")
df_hourly = get_granularity(data , "hourly")

# plot whole df_daily content
fig , axis = plt.subplots(figsize=(12,8))
axis.plot(df_daily.index , df_daily["feels_like"] , color = "tab:red" , label = "feels like")
axis.fill_between(df_daily.index , df_daily["min_temp"] , df_daily["max_temp"] , color = "tab:grey" , alpha = 0.25 , label = "temperature interval")
axis.set_xlabel("date")
axis.set_ylabel("temperature in °C")
axis.legend();

for i , row in df_daily.iterrows():
    print(f"{i.day}.{i.month}.{i.year}:\n\
        Highest temperature:{row['max_temp']}\n\
        Lowest temperature:{row['min_temp']}\n\
        Feels like:{row['feels_like']}\n\
        UV-Index:{row['uvi']}\n\
        Weather will be:")
    display(Image(filename=icons[row["weather"]] , width=250 , height=250))


# plot whole df_hourly DataFrame
# get local maximum
ymax = max(df_hourly["temp"])
xpos = np.where(df_hourly["temp"] == ymax)
xmax = df_hourly.index[xpos]

# create figure to plot
fig, axis = plt.subplots(figsize=(12, 8))
# add two plots for temperature and feels like
axis.plot(df_hourly.index, df_hourly["temp"], color="tab:red", label="actual temp")
axis.plot(df_hourly.index, df_hourly["feels_like"], color="tab:orange", label="feels like")
axis.set_title("Weather forecast per hour")
axis.set_ylabel("temperature in degree celsius")
axis.set_xlabel("date")
axis.legend(loc="upper right")
axis.annotate(
    f"{ymax} °C",
    xy=(xmax, ymax),
    xytext=(xmax, ymax + 0.25),
);


def get_weather_data_germany(
    api_key: str,
    shape_file: str,
) -> pd.DataFrame:
    """get geological data using shapefile and weather data using api request from openweather-api

    Args:
        api_key (str): necessary authentification to create api request
        shape_file (str): shape file for germany map used to plot with geopandas

    Returns:
        pd.DataFrame: prepared DataFrame
    """

    # read in shapefile and remove unnecessary columns
    germany = gpd.read_file(shape_file)
    germany = germany.drop(columns=["type", "id"])

    # add capitals for each federal state
    capital = [
        "Stuttgart",
        "Muenchen",
        "Berlin",
        "Potsdam",
        "Bremen",
        "Hamburg",
        "Wiesbaden",
        "Schwerin",
        "Hannover",
        "Düsseldorf",
        "Mainz",
        "Saarbrücken",
        "Magdeburg",
        "Dresden",
        "Kiel",
        "Erfurt",
    ]
    germany["capital"] = capital

    # get latitude and longitude for each capital and append it to DataFrame
    service = geopy.Nominatim(user_agent="myGeocoder")
    for i, row in germany.iterrows():
        location = service.geocode(f"{row['capital']}, Germany")
        germany.loc[i, "lat"] = location.latitude
        germany.loc[i, "lon"] = location.longitude

    # get current weather data for each capital in DataFrame
    for i, row in germany.iterrows():
        try:
            url = (
                "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric"
                % (row["lat"], row["lon"], api_key)
            )
            response = requests.get(url)
            data = json.loads(response.text)
        except Exception as e:
            print(e)

        # if there is a failue code in dataset continue with next city
        if "cod" in data.keys():
            print(row["name"], data["message"])
            continue

        # fill DataFrame with weather data from api request
        else:
            germany.loc[i, "temp"] = data["current"]["temp"]
            germany.loc[i, "feels_like"] = data["current"]["feels_like"]
            germany.loc[i, "weather_type"] = data["current"]["weather"][0]["main"]
            germany.loc[i, "pressure"] = data["current"]["pressure"]
            germany.loc[i, "uvi"] = data["current"]["uvi"]
            germany.loc[i, "clouds"] = data["current"]["clouds"]

    return germany


df_germany = get_weather_data_germany(API_KEY, SHAPE_FILE)
df_germany.plot(column="temp", legend=True, cmap="OrRd").set_axis_off()
