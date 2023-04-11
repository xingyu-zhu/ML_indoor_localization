from Data_Processing import get_data, data_processing, nonormalize_data
from sklearn_model import sklearn_MLP_model, sklearn_RF_model, MLP_model, RF_model
from tf_model import tf_KNN_model
from plot_output import draw_on_earth
import warnings
warnings.filterwarnings('ignore')

Train_data_file_path = './UjiIndoorLoc/TrainingData.csv'
Test_data_file_path = './UjiIndoorLoc/ValidationData.csv'

if __name__ == "__main__":
    df = get_data(train_data_path=Train_data_file_path, test_data_path=Test_data_file_path)
    feature_train, feature_test, target_train, target_test, feature_train_summary = data_processing(df)
    # feature_train, feature_test, target_train, target_test = nonormalize_data(df)
    # trained_model = tf_KNN_model(feature_train, target_train)
    # sklearn_MLP_model(feature_train, target_train, feature_test, target_test)
    # sklearn_RF_model(feature_train, target_train, feature_test, target_test)
    # predict_values, actual_values = MLP_model(feature_train, target_train, feature_test, target_test)
    predict_values, actual_values = RF_model(feature_train, target_train, feature_test, target_test)
    draw_on_earth(predict_values, actual_values)

