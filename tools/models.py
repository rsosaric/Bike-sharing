import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import tools.data_handling as dtools
import tools.plotting_tools as plotting


# Class inspired on https://thinkingneuron.com/bike-rental-demand-prediction-case-study-in-python/
class MLModel:
    def __init__(self, model_name: str, input_data: pd.DataFrame, settings):
        print("\n==>> Setting model: " + model_name)
        self.__model_name = model_name

        self.__model_base_estimator = None
        self.__is_decision_tree_model = False

        if self.__model_name == 'LinearRegression':
            self.__model = LinearRegression()

        elif self.__model_name == 'DecisionTreeRegressor':
            self.__is_decision_tree_model = True
            self.__model = DecisionTreeRegressor(max_depth=8, criterion='mse')
            # Good Range of Max_depth = 2 to 20

        elif self.__model_name == 'RandomForestRegressor':
            self.__is_decision_tree_model = True
            self.__model = RandomForestRegressor(max_depth=10, n_estimators=100, criterion='mse')
            # Good range for max_depth: 2-10 and n_estimators: 100-1000

        elif self.__model_name == 'AdaBoost':
            self.__is_decision_tree_model = True
            # Choosing Decision Tree with 1 level as the weak learner
            self.__model_base_estimator = DecisionTreeRegressor(max_depth=10)
            self.__model = AdaBoostRegressor(n_estimators=100, base_estimator=self.__model_base_estimator,
                                             learning_rate=0.04)
        elif self.__model_name == 'XGBoost':
            self.__is_decision_tree_model = True
            self.__model = XGBRegressor(max_depth=10,
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        objective='reg:squarederror',
                                        booster='gbtree')

        elif self.__model_name == 'KNN':
            self.__model = KNeighborsRegressor(n_neighbors=2)

        else:
            raise NotImplementedError("model not implemented")

        print(self.__model)

        self.__settings = settings
        self.__input_data = input_data
        self.__variables_for_training = self.__settings.variables_for_training
        self.__target_variable = self.__settings.target_variable
        self.__output_folder = self.__settings.output_folder_ml + self.__model_name + "/"
        dtools.check_and_create_folder(self.__output_folder)

        # Variables to be assigned using class methods
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
        training_summary_txt_lines = []
        self.prepare_train_test_data()
        print("Training model ...", end="")
        # Creating the model on Training Data
        self.__fitted_model = self.__model.fit(self.__x_train, self.__y_train)
        self.__prediction_from_x_test = self.__fitted_model.predict(self.__x_test)
        print(" (done)")

        # Measuring Goodness of fit in Training data
        print('     Final R2 Value:', metrics.r2_score(self.__y_train, self.__fitted_model.predict(self.__x_train)))
        print('-->> Model Validation and Accuracy Calculations: <<--')

        if self.__is_decision_tree_model:
            self.plot_feature_importance()

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
        training_summary_txt_lines.append('Mean Accuracy on test data:' + str(accuracy))
        print('Median Accuracy on test data:', median_accuracy)
        training_summary_txt_lines.append('Median Accuracy on test data:' + str(median_accuracy))

        # Custom Scoring MAPE calculation
        custom_scorer = make_scorer(MLModel.accuracy_score, greater_is_better=True)

        accuracy_values = cross_val_score(self.__model, self.__x_all, self.__y_all, cv=10, scoring=custom_scorer)
        print('\nAccuracy values for 10-fold Cross Validation:\n', accuracy_values)
        print('\nFinal Average Accuracy of the model:', round(accuracy_values.mean(), 2))
        training_summary_txt_lines.append('\nAccuracy values for 10-fold Cross Validation:\n' + str(accuracy_values))
        training_summary_txt_lines.append('\nFinal Average Accuracy of the model:' +
                                          str(round(accuracy_values.mean(), 2)))

        with open(self.__output_folder + "training_summary.txt", "w") as txt_file:
            for line in training_summary_txt_lines:
                txt_file.write(line + "\n")
        print("Summary about the training results has been stored in: " + self.__output_folder + "training_summary.txt")

    def plot_feature_importance(self):
        # Plotting the feature importance for Top 10 most important columns
        if self.__is_decision_tree_model:
            feature_importances = pd.Series(self.__model.feature_importances_, index=self.__variables_for_training)
            plot = feature_importances.nlargest(10).plot(kind='barh')
            plotting.save_py_fig_to_file(plot.get_figure(),
                                         self.__output_folder + "importances" + self.__settings.plot_extension,
                                         plot_dpi=self.__settings.plot_dpi)
        else:
            raise NotImplementedError("plot_feature_importance() not implemented for this model")

    @staticmethod
    def accuracy_score(real, prediction):
        mape_estimator = np.mean(100 * (np.abs(real - prediction) / real))
        return 100 - mape_estimator
