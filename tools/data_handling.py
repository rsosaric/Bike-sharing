import pandas as pd
import tools.plotting_tools as plotting
import numpy as np


# Import data as a pandas dataframe and print basic information
def read_dataset(file_path: str, print_data_summary: bool):
    print("reading input data from " + file_path + " ...")
    pd_data = pd.read_csv(file_path, encoding='latin')

    if print_data_summary:
        print("--------------- Input data summary --------------- ")
        print(pd_data)
        print(pd_data.info())
        print(pd_data.describe(include='all'))
        print(pd_data.nunique())

    return pd_data


# clean data from duplicates, exclude indicated columns, etc
def clean_data(pd_data: pd.DataFrame, columns_to_exclude: list = None):

    # Checking and removing duplicate rows
    pd_data.drop_duplicates(inplace=True)

    # Excluding columns
    if columns_to_exclude is not None:
        pd_data.drop(columns_to_exclude, axis=1, inplace=True)
        print("\n Columns " + str(columns_to_exclude) + " were excluded. \n")


def input_data_analysis(pd_data: pd.DataFrame, output_folder: str):
    categorical_variables = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    continuous_variables = ['temp', 'atemp', 'hum', 'windspeed', 'registered']
    columns_in_df = list(pd_data)
    categorical_variables = list(set(categorical_variables).intersection(columns_in_df))
    continuous_variables = list(set(continuous_variables).intersection(columns_in_df))

    plotting.plot_several_hists_from_data_frame(pd_data, col_labels=continuous_variables,
                                                output_file=output_folder+"continuous_variables_hists.pdf")



