import tools.data_handling as dtools
import tools.plotting_tools as plotting
import tools.models as models

# Settings
columns_to_exclude_from_model = ['yr']
output_folder_data_analysis = "outputs/data_analysis/"
output_folder_ml = "outputs/ml/"

input_data = dtools.read_dataset("Bike-Sharing-Dataset/hour.csv", True)
dtools.clean_data(input_data, columns_to_exclude_from_model)

plotting.check_and_create_folder(output_folder_data_analysis)
plotting.check_and_create_folder(output_folder_ml)

dtools.input_data_analysis(input_data, output_folder_data_analysis)



