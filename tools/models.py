from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import tools.data_handling as dtools


# Class inspired on https://thinkingneuron.com/bike-rental-demand-prediction-case-study-in-python/
class MLModel:
    def __init__(self, model_name: str, input_data: pd.DataFrame, variables_for_training: list, target_variable: str):
        print("\n==>> Setting model: " + model_name)
        self.__model_name = model_name

        if self.__model_name == 'LinearRegression':
            self.__model = LinearRegression()
        elif self.__model_name == 'DecisionTreeRegressor':
            self.__model = DecisionTreeRegressor(max_depth=8, criterion='mse')
        else:
            raise AssertionError("model not implemented")

        self.__input_data = input_data
        self.__variables_for_training = variables_for_training
        self.__target_variable = target_variable
        self.__x_all = None
        self.__y_all = None
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
        self.__fitted_model = None
        self.__prediction_from_x_test = None

    def prepare_train_test_data(self):
        self.__x_all, self.__y_all, self.__x_train, self.__x_test, self.__y_train, self.__y_test = \
            dtools.get_ml_data(self.__input_data)

    def train_model(self):
        print("Training model ...", end="")
        self.prepare_train_test_data()
        # Creating the model on Training Data
        self.__fitted_model = self.__model.fit(self.__x_train, self.__y_train)
        self.__prediction_from_x_test = self.__fitted_model.predict(self.__x_test)
        print(" (done)")

        # Measuring Goodness of fit in Training data
        print('     Final R2 Value:', metrics.r2_score(self.__y_train, self.__fitted_model.predict(self.__x_train)))
        print('-->> Model Validation and Accuracy Calculations: <<--')

        # Generating dataframe for testing data results
        testing_data_results = pd.DataFrame(data=self.__x_test, columns=self.__variables_for_training)
        testing_data_results[self.__target_variable] = self.__y_test
        predicted_target_variable_name = 'Predicted ' + self.__target_variable
        testing_data_results[predicted_target_variable_name] = np.round(self.__prediction_from_x_test)
        # print(testing_data_results[[self.__target_variable, predicted_target_variable_name]].head())

        # Calculating the error for each row
        ape_column_name = 'APE'
        testing_data_results[ape_column_name] = \
            100 * ((abs(testing_data_results[self.__target_variable] -
                        testing_data_results[predicted_target_variable_name])) / testing_data_results[self.__target_variable])
        accuracy = 100 - np.mean(testing_data_results[ape_column_name])
        median_accuracy = 100 - np.median(testing_data_results[ape_column_name])
        print('Mean Accuracy on test data:', accuracy)
        print('Median Accuracy on test data:', median_accuracy)

        # Custom Scoring MAPE calculation
        custom_scorer = make_scorer(MLModel.accuracy_score, greater_is_better=True)

        accuracy_values = cross_val_score(self.__model, self.__x_all, self.__y_all, cv=10, scoring=custom_scorer)
        print('\nAccuracy values for 10-fold Cross Validation:\n', accuracy_values)
        print('\nFinal Average Accuracy of the model:', round(accuracy_values.mean(), 2))

    @staticmethod
    def accuracy_score(real, prediction):
        mape_estimator = np.mean(100 * (np.abs(real - prediction) / real))
        return 100 - mape_estimator
