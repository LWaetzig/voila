import datetime as dt
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import requests
import random
import geopy

os.getcwd()
os.chdir("..")


api_key = "0903f43c667c445d4a0c16920ef81c36"
lat = "51.050407"
lon = "13.737262"

url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric"%(lat, lon, api_key)

response = requests.get(url)
data = json.loads(response.text)

hourly = data["hourly"]

weather_hourly = pd.DataFrame()

datetimes = list()
temp = list()
feels_like = list()

for entry in hourly:
    datetimes.append(entry["dt"])
    temp.append(entry["temp"])
    feels_like.append(entry["feels_like"])

weather_hourly["dt"] = datetimes
weather_hourly["temp"] = temp
weather_hourly["feels_like"] = feels_like


for i in range(len(weather_hourly)):
    weather_hourly["date"] = dt.datetime.fromordinal(weather_hourly.loc[i , "dt"])


date = list()
for entry in hourly:
    date.append(dt.fromtimestamp(entry["dt"], pytz.timezone('Europe/Vienna')))


weather_hourly["dt"] = date
weather_hourly = weather_hourly.set_index(weather_hourly["dt"])
weather_hourly = weather_hourly[["temp","feels_like"]]

plt.plot(weather_hourly.index , weather_hourly["temp"])


germany = gpd.read_file("data/germany_geo.json")

germany = germany.drop(columns=["type" , "id"])

temp = [random.randint(0,40) for i in range(16)]
germany["temp"] = temp

germany.plot(column="temp" , legend = True , cmap="OrRd").set_axis_off()


service = geopy.Nominatim(user_agent = "myGeocoder")
for i , row in germany.iterrows():
    location = service.geocode(f"{row['name']}, Germany")
    germany.loc[i,"lat"] = location.latitude
    germany.loc[i , "lon"] = location.longitude

