import pandas as pd
import math
from sklearn.model_selection import train_test_split
import shap
import numpy as np

def get_data(train_data_path, test_data_path):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    df = pd.concat([df_train, df_test])

    df.drop(columns=["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"], inplace=True)

    return df

def normalize(x, xmin, xmax, a, b):
    numerator = x - xmin
    denominator = xmax - xmin
    multiplier = b - a
    ans = (numerator / denominator) * multiplier + a

    return ans

def normalize_wifi(num):
    sig_min = -104
    sig_max = 0
    tar_min = 0.25
    tar_max = 1.0
    no_sig = 100

    ans = 0
    num = float(num)
    if math.isclose(num, no_sig, rel_tol=1e-3):
        return 0
    else:
        ans = normalize(num, sig_min, sig_max, tar_min, tar_max)
        return ans

def normalize_lat(num):
    lat_min = 4864745.7450159714
    lat_max = 4865017.3646842018
    tar_min = 0
    tar_max = 1

    num = float(num)
    ans = normalize(num, lat_min, lat_max, tar_min, tar_max)

    return ans

def normalize_long(num):
    long_min = -7695.9387549299299000
    long_max = -7299.786516730871000
    tar_min = 0
    tar_max = 1

    num = float(num)
    ans = normalize(num, long_min, long_max, tar_min, tar_max)

    return ans

def data_processing(dataframe):
    wifi_cells = dataframe.columns[:520]

    for index in wifi_cells:
        dataframe[index] = dataframe[index].apply(normalize_wifi)

    dataframe["LATITUDE"] = dataframe["LATITUDE"].apply(normalize_lat)
    dataframe["LONGITUDE"] = dataframe["LONGITUDE"].apply(normalize_long)

    feature_values = dataframe[wifi_cells]
    target_values = dataframe[["LATITUDE", "LONGITUDE", "BUILDINGID", "FLOOR"]]

    feature_train, feature_test, target_train, target_test = train_test_split(feature_values, target_values,
                                                                              random_state=0, test_size=0.15)

    feature_train_summary = shap.kmeans(feature_train, 10)

    return feature_train, feature_test, target_train, target_test, feature_train_summary

def denormalize(ans, xmin, xmax, a, b):
    x = (ans - a) * (xmax - xmin) / (b - a) + xmin

    return x

def denormalize_lat(num):
    lat_min = 4864745.7450159714
    lat_max = 4865017.3646842018
    tar_min = 0
    tar_max = 1

    num = float(num)
    x = denormalize(num, lat_min, lat_max, tar_min, tar_max)

    return x

def denormalize_long(num):
    long_min = -7695.9387549299299000
    long_max = -7299.786516730871000
    tar_min = 0
    tar_max = 1

    num = float(num)
    x = denormalize(num, long_min, long_max, tar_min, tar_max)

    return x

def nonormalize_data(dataframe):
    wifi_cells = dataframe.columns[:520]
    feature_values = dataframe[wifi_cells]
    target_values = dataframe[["LATITUDE", "LONGITUDE", "BUILDINGID", "FLOOR"]]

    feature_train, feature_test, target_train, target_test = train_test_split(feature_values, target_values,
                                                                              random_state=0, test_size=0.15)

    return feature_train, feature_test, target_train, target_test

