default_training_data = "Bike-Sharing-Dataset/hour.csv"
columns_to_exclude_from_model = ['yr']
categorical_variables = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
continuous_variables = ['temp', 'atemp', 'hum', 'windspeed', 'registered']
do_input_data_analysis = False  # Produce plots and correlation studies for the input data
do_extra_train_model_studies = False  # Produce analysis of model training performance
force_training = False  # If True: re-train model even if the final model outputs are available

target_variable = 'cnt'
# variables_for_training = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'registered']
variables_for_training = ['registered', 'mnth', 'hr', 'weekday']
test_sample_size = 0.3
# ml_model = 'LinearRegression'
# ml_model = 'KNN'
# ml_model = 'DecisionTreeRegressor'
# ml_model = 'RandomForestRegressor'
# ml_model = 'AdaBoost'
ml_model = 'XGBoost'

output_folder_data_analysis = "outputs/data_analysis/"
output_folder_ml = "outputs/ml/"

plot_dpi = None
plot_extension = ".png"
