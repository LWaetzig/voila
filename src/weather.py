os.getcwd()
os.chdir("..")

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

icons = {
    "Clear" : "figures/sun.png",
    "Rain" : "figures/rain.png",
    "sun_cloud" : "figures/sun_cloud.png",
    "Clouds" : "figures/cloud.png",
    "wind" : "figures/wind.png",
}

# define global vairables
API_KEY = "0903f43c667c445d4a0c16920ef81c36"
SHAPE_FILE = "data/germany_geo.json"

# input for city
# city_name = str(input("give me a city name: "))



def get_weather_data_city(
    city_name: str,
    api_key: str
)-> pd.DataFrame:
    """function to get weather data for one specific city

    Args:
        city_name (str): name of city in germany
        api_key (str): necessary authentification to create api request

    Returns:
        pd.DataFrame: filled DataFrame with weather data from api
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
        # only select hourly data from api request
        forecast = data["hourly"]

        # create, fill DataFrame for weather data and set timestamps as index
        weather_df = pd.DataFrame()
        weather_df["date"] = [
            dt.datetime.fromtimestamp(entry["dt"], pytz.timezone("Europe/Berlin"))
            for entry in forecast
        ]
        weather_df["temp"] = [entry["temp"] for entry in forecast]
        weather_df["feels_like"] = [entry["feels_like"] for entry in forecast]
        weather_df["pressure"] = [entry["pressure"] for entry in forecast]
        weather_df["uvi"] = [entry["uvi"] for entry in forecast]
        weather_df["clouds"] = [entry["clouds"] for entry in forecast]
        weather_df["weather"] = [entry["weather"][0]["main"] for entry in forecast]
        weather_df = weather_df.set_index("date", drop=True)

        return weather_df


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


df = get_weather_data_city(city_name="Dresden" , api_key=API_KEY)

# get local maximum 
ymax = max(df["temp"])
xpos = np.where(df["temp"] == ymax)
xmax = df.index[xpos]

# create figure to plot
fig , axes = plt.subplots(2 , figsize=(12,8))
# add to plots for temperature and feels like
axes[0].plot(df.index , df["temp"] , color = "tab:red" , label = "actual temp")
axes[0].plot(df.index , df["feels_like"] , color = "tab:orange" , label = "feels like")
axes[0].set_title("Weather forecast per hour")
axes[0].set_ylabel("temperature in degree celsius")
axes[0].set_xlabel("date")
axes[0].legend(loc="upper right");
axes[0].annotate(f"{ymax} °C", xy=(xmax, ymax), xytext=(xmax, ymax + 0.25),);


# plt.plot(weather_hourly.index, weather_hourly["temp"])


fig , axes = plt.subplots(5,1)
axes.ravel()
helper = 0
for key , value in icons.items():
    axes[helper].imshow(mpimg.imread(value))
    axes[helper].axis("off")
    helper += 1

fig.tight_layout()


corona = pd.read_csv("data/corona.csv")