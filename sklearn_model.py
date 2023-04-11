from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, learning_curve, LearningCurveDisplay
from sklearn.svm import SVC
import shap
from shap import summary_plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import neural_network as nn
from math import sqrt

def sklearn_MLP_model(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)

    MLP_model = nn.MLPRegressor(hidden_layer_sizes=(300, 100, 4), activation='relu')
    trained_model = MLP_model.fit(feature_train, target_train)

    R2_train_score = MLP_model.score(feature_train, target_train)
    print("MLP R2 Scores: " + str(R2_train_score))

    cross_val_RMSE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    print("Cross validation average RMSE scores: " + str(Average_RMSE_score))

    cross_val_MAE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    print("Cross validation average MAE scores: " + str(Average_MAE_score))

    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)
    R2_test_score = MLP_model.score(feature_test, target_test)
    print("MLP R2 test Scores:" + str(R2_test_score))

    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))

    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))

    return MLP_model.predict, train_predict, test_predict, trained_model

def sklearn_RF_model(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    RF_model = RandomForestRegressor(n_estimators=500)
    trained_model = RF_model.fit(feature_train, target_train)
    R2_train_score = RF_model.score(feature_train, target_train)
    print("RF R2 Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    print("Cross validation average RMSE scores: " + str(Average_RMSE_score))
    cross_val_MAE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    print("Cross validation average MAE scores: " + str(Average_MAE_score))
    R2_test_score = RF_model.score(feature_test, target_test)
    print("RF R2 test Scores:" + str(R2_test_score))
    train_predict = RF_model.predict(feature_train)
    test_predict = RF_model.predict(feature_test)
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))

    return RF_model.predict, train_predict, test_predict

def MLP_model(feature_train, target_train, feature_test, target_test):
    MLP_model = nn.MLPRegressor(hidden_layer_sizes=(300, 100, 4), activation='relu')
    trained_model = MLP_model.fit(feature_train, target_train)

    R2_train_score = MLP_model.score(feature_train, target_train)
    print("MLP R2 Scores: " + str(R2_train_score))

    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)

    target_train_plot = target_train.head(50)
    train_predict_plot = train_predict[:50, :]

    target_test_plot = target_test.head(50)
    # print(target_test_plot)
    test_predict_plot = test_predict[:50, :]
    # print(test_predict_plot)

    # print(test_predict_plot[:, 0])
    # print(target_test_plot["LATITUDE"])

    test_error = 0
    cal_lat = target_test["LATITUDE"].values
    cal_lon = target_test["LONGITUDE"].values

    for i in range(len(test_predict)):
        test_error += sqrt((test_predict[i, 0] - cal_lat[i]) * (test_predict[i, 0] - cal_lat[i])
                           + (test_predict[i, 1] - cal_lon[i]) * (test_predict[i, 1] - cal_lon[i]))
    test_error /= len(test_predict)
    print(test_error)

    return train_predict_plot, target_train_plot

def RF_model(feature_train, target_train, feature_test, target_test):
    RF_model = RandomForestRegressor(n_estimators=500)
    trained_model = RF_model.fit(feature_train, target_train)

    R2_train_score = RF_model.score(feature_train, target_train)
    print("RF R2 Scores: " + str(R2_train_score))

    train_predict = RF_model.predict(feature_train)
    test_predict = RF_model.predict(feature_test)

    target_test_plot = target_test.head(50)
    # print(target_test_plot)
    test_predict_plot = test_predict[:50, :]
    # print(test_predict_plot)

    # print(test_predict_plot[:, 0])
    # print(target_test_plot["LATITUDE"])

    test_error = 0
    cal_lat = target_test["LATITUDE"].values
    cal_lon = target_test["LONGITUDE"].values

    for i in range(len(test_predict)):
        test_error += sqrt((test_predict[i, 0] - cal_lat[i]) * (test_predict[i, 0] - cal_lat[i])
                           + (test_predict[i, 1] - cal_lon[i]) * (test_predict[i, 1] - cal_lon[i]))
    test_error /= len(test_predict)
    print(test_error)

    return test_predict_plot, target_test_plot






