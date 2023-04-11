import numpy as np
import utm

from Data_Processing import denormalize_lat, denormalize_long
from matplotlib import pyplot as plt
import gif
from pyproj import Transformer


def get_lat_long_values(predicted_values, actual_values):
    latitude_predict = []
    longitude_predict = []

    for lat in range(len(predicted_values)):
        denormalize_lat_value = denormalize_lat(predicted_values[lat, 0])
        latitude_predict.append(denormalize_lat_value)

    for long in range(len(predicted_values)):
        denormalize_long_value = denormalize_long(predicted_values[long, 1])
        longitude_predict.append(denormalize_long_value)

    actual_values["LATITUDE"] = actual_values["LATITUDE"].apply(denormalize_lat)
    actual_values["LONGITUDE"] = actual_values["LONGITUDE"].apply(denormalize_long)

    latitude_actual = actual_values["LATITUDE"].values
    longitude_actual = actual_values["LONGITUDE"].values

    return latitude_predict, longitude_predict, latitude_actual, longitude_actual

@gif.frame
def plot(i, lon_pre, lat_pre, lon_act, lat_act):
    x_actual = lon_act[i]
    y_actual = lat_act[i]
    actual = plt.scatter(x_actual, y_actual)

    x_predict = lon_pre[i]
    y_predict = lat_pre[i]
    predict = plt.scatter(x_predict, y_predict)

    plt.xlim((-2.92, -2.89))
    plt.ylim((43.92, 43.96))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.legend(handles=[predict, actual], labels=['Predict Position', 'Actual Position'], loc='upper right')

def draw_on_earth(predicted_values, actual_values):
    latitude_predict, longitude_predict, latitude_actual, longitude_actual = get_lat_long_values(predicted_values, actual_values)

    transformer = Transformer.from_crs("epsg:32730", "epsg:4326")
    print(latitude_predict)
    print(longitude_predict)

    predict_lat = []
    predict_long = []

    actual_lat = []
    actual_long = []

    for num in range(len(latitude_predict)):
        x = latitude_predict[num]
        y = longitude_predict[num]
        lat, long = utm.to_latlon(500000 - y, x, 30, 'S')
        predict_lat.append(lat)
        predict_long.append(long)

    for num in range(len(latitude_actual)):
        x = latitude_actual[num]
        y = longitude_actual[num]

        lat, long = utm.to_latlon(500000 - y, x, 30, 'S')
        actual_lat.append(lat)
        actual_long.append(long)

    print(predict_long)
    print(predict_lat)

    gif.options.matplotlib["dpi"] = 300

    frames = []

    for i in range(len(latitude_predict)):
        frame = plot(i, predict_long, predict_lat, actual_long, actual_lat)
        frames.append(frame)

    gif.save(frames, './result.gif', duration=120)






