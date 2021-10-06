default_training_data = "Bike-Sharing-Dataset/hour.csv"  # Path to training data
columns_to_exclude_from_model = ['yr']  # Columns to exclude from data analysis
output_folder_data_analysis = "outputs/data_analysis/"  # Output folder for data analysis plots
output_folder_ml = "outputs/ml/"  # Output folder for ML results including output files for the final model
plot_dpi = None  # plot image quality
plot_extension = ".png"  # plot output format

categorical_variables = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']  # Name of the columns in data containing categorical variables
continuous_variables = ['temp', 'atemp', 'hum', 'windspeed', 'registered']  # Name of the columns in data containing continuous variables

do_input_data_analysis = False  # Produce plots and correlation studies for the input data
do_extra_train_model_studies = False  # Produce analysis of model training performance
force_training = False  # If True: re-train model even if the final model outputs are available

target_variable = 'cnt'  # variable to be predicted
variables_for_training = ['registered', 'season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
variables_for_training = ['registered', 'hr', 'mnth', 'workingday', 'holiday']  # variables used for training
test_sample_size = 0.3  # part of data to be used as test data

# ** Available models (also check in models.py) **
# ml_model = 'LinearRegression'
# ml_model = 'KNN'
# ml_model = 'DecisionTreeRegressor'
# ml_model = 'RandomForestRegressor'
# ml_model = 'AdaBoost'
ml_model = 'XGBoost'  # ML model to use
