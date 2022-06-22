import datetime as dt
import json
import os

import geopandas as gpd
import pandas as pd
import pytz
import requests
import geopy

os.getcwd()
os.chdir("..")


api_key = "0903f43c667c445d4a0c16920ef81c36"
shape_file = "data/germany_geo.json"
city_name = "Dresden"

service = geopy.Nominatim(user_agent="myGeocoder")
Location = service.geocode(f"{city_name}, Germany")

try:
    url = (
            "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric"
            % (Location.latitude, Location.longitude, api_key)
        )
    response = requests.get(url)
    data = json.loads(response.text)

except Exception as e:
    print(f"something went wrong: {e}")

if "cod" in data.keys():
    print(data["cod"] , data["message"])

else:
    # only select hourly data from api request
    hourly = data["hourly"]

    weather_hourly = pd.DataFrame()

    for entry in hourly:
        dt = entry["dt"]
        date = dt.datetime.fromordinal(dt)
        print(date)




def setup_dataframe(api_key : str , shape_file : str , ) -> pd.DataFrame:
    """setup dataframe with geo-data for germany and weather data from openweather-api

    Args:
        api_key (str): necessary authentification to use api request
        shape_file (str): shape file for germany map used to plot geopandas

    Returns:
        pd.DataFrame: prepared DataFrame
    """

    # read in shape file and remove unnecessary columns
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

        if "cod" in data.keys():
            print(row["name"], data["message"])
            continue

        else:
            germany.loc[i, "temp"] = data["current"]["temp"]
            germany.loc[i, "feels_like"] = data["current"]["feels_like"]
            germany.loc[i, "weather_type"] = data["current"]["weather"][0]["main"]
            germany.loc[i, "pressure"] = data["current"]["pressure"]
            germany.loc[i, "uvi"] = data["current"]["uvi"]
            germany.loc[i, "clouds"] = data["current"]["clouds"]

    return germany


df = setup_dataframe(api_key , shape_file)


# plt.plot(weather_hourly.index, weather_hourly["temp"])
df.plot(column="temp", legend=True, cmap="OrRd").set_axis_off()

