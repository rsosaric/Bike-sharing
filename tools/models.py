import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import pickle
import os
import settings
import tools.data_handling as dtools
import tools.plotting_tools as plotting


# Class inspired on https://thinkingneuron.com/bike-rental-demand-prediction-case-study-in-python/
class MLModel:
    def __init__(self, model_name: str, settings):
        print("\n==>> Setting model: " + model_name)
        self.__model_name = model_name

        self.__model_base_estimator = None
        self.__is_decision_tree_model = False
        self.__do_data_standardization = True

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

        self.__predictor_scaler = MinMaxScaler()

        self.__do_extra_train_model_studies = settings.do_extra_train_model_studies
        if self.__do_extra_train_model_studies:
            print(self.__model)

        self.__settings = settings
        self.__columns_to_exclude_from_model = settings.columns_to_exclude_from_model
        self.__output_folder_for_data_analysis = settings.output_folder_data_analysis
        self.__do_input_data_analysis = settings.do_input_data_analysis
        self.__path_for_training_data = settings.default_training_data
        self.__variables_for_training = self.__settings.variables_for_training
        self.__target_variable = self.__settings.target_variable
        self.__output_folder = self.__settings.output_folder_ml + self.__model_name + "/"
        self.__test_sample_size = settings.test_sample_size

        self.__reset_training = settings.force_training
        if self.__reset_training:
            print("** INFO: Re-training model (set in settings)")

        dtools.check_and_create_folder(self.__output_folder)

        self.__saved_ml_data_file_pkl = self.__output_folder + 'ml_data.pkl'
        self.__ml_data = None
        self.__final_model_saved_file_pkl = self.__output_folder + 'final_model.pkl'
        # TODO: store also in Predictive Model Markup Language (PMML) format
        self.__final_model = None
        self.__saved_predictor_scale_fit_pkl = self.__output_folder + 'predictor_scale.pkl'
        self.__predictor_scaler_fit = None

        # Variables to be assigned using class methods
        self.__x_all = None
        self.__y_all = None
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
        self.__fitted_model = None
        self.__prediction_from_x_test = None
        self.__input_training_data = None

    # Get data ready for its use in ML: select columns, convert to numerical values, etc
    def get_ml_data(self):

        # selecting the data for ML
        print("** INFO: Using only data from " + str(self.__variables_for_training))

        self.__input_training_data = dtools.read_dataset(self.__path_for_training_data, print_data_summary=False)
        dtools.clean_data(self.__input_training_data, columns_to_exclude=self.__columns_to_exclude_from_model)

        if self.__do_input_data_analysis:
            dtools.input_data_analysis(self.__input_training_data, output_folder=self.__output_folder_for_data_analysis,
                                       plot_extension=settings.plot_extension,
                                       target_variable=self.__target_variable,
                                       categorical_variables=settings.categorical_variables,
                                       continuous_variables=settings.continuous_variables)
        self.__ml_data = self.__input_training_data[self.__variables_for_training]
        # Saving selected data for further use
        self.__ml_data.to_pickle(self.__saved_ml_data_file_pkl)

        # Treating all the nominal variables at once using dummy variables
        numeric_ml_data = pd.get_dummies(self.__ml_data)

        # Adding Target Variable to the data
        numeric_ml_data[self.__target_variable] = self.__input_training_data[self.__target_variable]

        # Split the data into training and testing set
        self.__x_all = numeric_ml_data[self.__variables_for_training].values
        self.__y_all = numeric_ml_data[self.__target_variable].values

        if self.__do_data_standardization:
            self.__x_all = self.data_standardization(self.__x_all)

        if self.__do_extra_train_model_studies:
            print("** INFO: Using test size = " + str(self.__test_sample_size))
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = \
                train_test_split(self.__x_all, self.__y_all, test_size=self.__test_sample_size, random_state=12)

    def train_test_model(self):
        training_summary_txt_lines = []
        if self.__do_extra_train_model_studies:
            print("Training model (analysis mode) ...", end="")
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

        print("\nGetting model accuracy ...", end="")
        # Custom Scoring MAPE calculation
        custom_scorer = make_scorer(MLModel.accuracy_score, greater_is_better=True)

        accuracy_values = cross_val_score(self.__model, self.__x_all, self.__y_all, cv=10, scoring=custom_scorer)
        print(" (done)")
        if self.__do_extra_train_model_studies:
            print('Accuracy values for 10-fold Cross Validation:\n', accuracy_values)
            training_summary_txt_lines.append(
                'Accuracy values for 10-fold Cross Validation:\n' + str(accuracy_values))
        print('Final Average Accuracy of the model:', round(accuracy_values.mean(), 2))
        training_summary_txt_lines.append('\nFinal Average Accuracy of the model:' +
                                          str(round(accuracy_values.mean(), 2)))

        with open(self.__output_folder + "training_summary.txt", "w") as txt_file:
            for line in training_summary_txt_lines:
                txt_file.write(line + "\n")
            txt_file.close()
        print("Summary about the training results has been stored in: " + self.__output_folder + "training_summary.txt")

    def train_with_all_data(self):
        self.get_ml_data()
        self.train_test_model()
        print("\nTraining model ...", end="")
        self.__final_model = self.__model.fit(self.__x_all, self.__y_all)
        print(" (done)")
        # Saving the final model
        with open(self.__final_model_saved_file_pkl, 'wb') as fileWriteStream:
            pickle.dump(self.__final_model, fileWriteStream)
            fileWriteStream.close()
        print('Predictive Model is saved at ' + self.__final_model_saved_file_pkl)

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

    def predict(self, input_data: pd.DataFrame):
        exists_saved_ml_data_file_pkl = os.path.isfile(self.__saved_ml_data_file_pkl)
        exists_final_model_saved_file_pkl = os.path.isfile(self.__final_model_saved_file_pkl)

        if self.__reset_training or not exists_saved_ml_data_file_pkl or not exists_final_model_saved_file_pkl:
            if self.__reset_training:
                print("\n Starting model training (reset_training is enabled)")
            else:
                print("\n Starting model training <- Some training output files not found:")
                if not exists_saved_ml_data_file_pkl:
                    print("     -" + self.__saved_ml_data_file_pkl)
                if not exists_final_model_saved_file_pkl:
                    print("     -" + self.__final_model_saved_file_pkl)
                print("\n")

            self.train_with_all_data()

        exists_saved_ml_data_file_pkl = os.path.isfile(self.__saved_ml_data_file_pkl)
        exists_final_model_saved_file_pkl = os.path.isfile(self.__final_model_saved_file_pkl)

        if not exists_saved_ml_data_file_pkl or not exists_final_model_saved_file_pkl:
            raise AssertionError("Something went wrong in the training and the "
                                 ".pkl for data and model were not produced!!")

        # Appending the new data with the Training data in order to ensure compatible format
        # TODO: Improve this implementation -> perform a dedicated format check
        self.__ml_data = pd.read_pickle(self.__saved_ml_data_file_pkl)
        n_inputs = input_data.shape[0]
        input_data = input_data.append(self.__ml_data)
        input_data = pd.get_dummies(input_data)

        # Generating the input values to the model
        x_input = input_data[self.__variables_for_training].values[0:n_inputs]

        # Input data standardization/normalization if needed (same normalization as as for training)
        if self.__do_data_standardization:
            if self.__predictor_scaler_fit is None:
                if os.path.isfile(self.__saved_predictor_scale_fit_pkl):
                    with open(self.__saved_predictor_scale_fit_pkl, 'rb') as fileReadStream:
                        self.__predictor_scaler_fit = pickle.load(fileReadStream)
                        fileReadStream.close()
                else:
                    raise IOError("File not found: " + self.__saved_predictor_scale_fit_pkl)

            x_input = self.__predictor_scaler_fit.transform(x_input)

        # Loading final model
        with open(self.__final_model_saved_file_pkl, 'rb') as fileReadStream:
            self.__final_model = pickle.load(fileReadStream)
            fileReadStream.close()

        # Get cnt Predictions
        prediction = self.__final_model.predict(x_input)
        prediction_result = pd.DataFrame(prediction, columns=['Prediction'])

        print("\nPredicted values:")
        print(prediction_result)
        print("\n")
        return round(prediction_result)

    # take inputs and return prediction
    def generate_prediction_from_values(self, input_registered, input_month, input_hour, input_weekday):
        # Creating a data frame for the model input
        if type(input_registered) == list:
            input_data = pd.DataFrame(
                data=list(zip(input_registered, input_month, input_hour, input_weekday)),
                columns=['registered', 'mnth', 'hr', 'weekday'])
        else:
            input_data = pd.DataFrame(
                data=[[input_registered, input_month, input_hour, input_weekday]],
                columns=['registered', 'mnth', 'hr', 'weekday'])

        predictions = self.predict(input_data=input_data)

        # Returning the predictions
        return predictions.to_json()

    def data_standardization(self, in_data: pd.DataFrame):
        # Standardization/Normalization of data
        self.__predictor_scaler_fit = self.__predictor_scaler.fit(in_data)
        with open(self.__saved_predictor_scale_fit_pkl, 'wb') as fileWriteStream:
            pickle.dump(self.__predictor_scaler_fit, fileWriteStream)
            fileWriteStream.close()
        # Generating the standardized values of X
        return self.__predictor_scaler_fit.transform(in_data)

    @staticmethod
    def accuracy_score(real, prediction):
        mape_estimator = np.mean(100 * (np.abs(real - prediction) / real))
        return 100 - mape_estimator
