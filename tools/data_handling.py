import pandas as pd
import os
import tools.plotting_tools as plotting


# Import data as a pandas dataframe and print basic information
def read_dataset(file_path: str, print_data_summary: bool):
    print("reading input data from " + file_path + " ...", end="")
    pd_data = pd.read_csv(file_path, encoding='latin')
    print(" (done)")
    print(pd_data.info())

    if print_data_summary:
        print("--------------- Input data summary --------------- ")
        print(pd_data)
        print(pd_data.describe(include='all'))
        print(pd_data.nunique())

    return pd_data


# clean data from duplicates, exclude indicated columns, etc
def clean_data(pd_data: pd.DataFrame, columns_to_exclude: list):
    # Checking and removing duplicate rows
    pd_data.drop_duplicates(inplace=True)

    # Excluding columns
    if columns_to_exclude is not None:
        pd_data.drop(columns_to_exclude, axis=1, inplace=True)
        print("\n** INFO: Columns " + str(columns_to_exclude) + " were excluded.")

    # Finding how many missing values are there for each column
    null_col_information = pd_data.isnull().sum()

    # Drop nan values from the input data
    all_data_size = len(pd_data)
    pd_data.dropna(inplace=True)
    no_nan_data_size = len(pd_data)
    remaining_data_rel = no_nan_data_size/all_data_size
    if remaining_data_rel < 0.95:
        print("A considerable amount of data has been excluded due to NaN values: " +
              str((1 - remaining_data_rel)*100) + "%")
        print(null_col_information)


def input_data_analysis(pd_data: pd.DataFrame, output_folder: str, target_variable: str, plot_extension: str,
                        categorical_variables: list, continuous_variables: list):
    variables_distribution_output_folder = output_folder + "variables_data_distribution/"
    check_and_create_folder(variables_distribution_output_folder)

    columns_in_df = list(pd_data)
    categorical_variables = list(set(categorical_variables).intersection(columns_in_df))
    continuous_variables = list(set(continuous_variables).intersection(columns_in_df))

    plotting.plot_hist_from_data_frame(pd_data, col_label=target_variable,
                                       output_file=variables_distribution_output_folder + target_variable + plot_extension)

    for i_var in categorical_variables:
        count_plot_name = i_var + plot_extension
        print("plotting " + count_plot_name)
        plotting.plot_categorical_data_distribution(pd_data, col_label=i_var,
                                                    output_file=variables_distribution_output_folder + count_plot_name)

        plot_name = i_var + "_vs_" + target_variable + plot_extension
        print("plotting " + plot_name)
        plotting.plot_categorical_data_vs_reference_variable(pd_data, col_label=i_var,
                                                             reference_variable=target_variable,
                                                             plot_type="points",
                                                             output_file=variables_distribution_output_folder + plot_name)

        plot_name = i_var + "_vs_" + target_variable + '_box' + plot_extension
        print("plotting " + plot_name)
        plotting.plot_categorical_data_vs_reference_variable(pd_data, col_label=i_var,
                                                             reference_variable=target_variable,
                                                             plot_type="box",
                                                             output_file=variables_distribution_output_folder + plot_name)

    # Plotting of the variables distributions
    plotting.plot_several_hists_from_data_frame(pd_data, col_labels=continuous_variables,
                                                output_file=variables_distribution_output_folder + "continuous_variables_hists" + plot_extension)
    for i_var in continuous_variables:
        plot_name = i_var + "_vs_" + target_variable + '_kde' + plot_extension
        print("plotting " + plot_name)
        plotting.plot_continuous_data_vs_reference_variable(pd_data, col_label=i_var,
                                                            reference_variable=target_variable,
                                                            plot_type='kde',
                                                            output_file=variables_distribution_output_folder + plot_name)

        plot_name = i_var + "_vs_" + target_variable + '_2d_map' + plot_extension
        print("plotting " + plot_name)
        plotting.plot_continuous_data_vs_reference_variable(pd_data, col_label=i_var,
                                                            reference_variable=target_variable,
                                                            plot_type='2d_map',
                                                            output_file=variables_distribution_output_folder + plot_name)

    # Relationship exploration (Categorical Variables)
    print("\n     *******  Relationship exploration (Categorical Variables)(ANOVA Results)  ******* \n")
    anova_function(pd_data=pd_data, target_variable=target_variable,
                   categorical_predictor_list=categorical_variables)

    # Relationship exploration (Continuous Variables)
    print("\n\n    *******  Relationship exploration (Categorical Variables)  ******* \n")
    correlation_results = pd_data[continuous_variables + [target_variable]].corr()
    print(pd_data[continuous_variables].corr())
    print("Selected variables:")
    print(correlation_results[target_variable][abs(correlation_results[target_variable]) > 0.5])


# Defining a function to find the statistical relationship with all the categorical variables
def anova_function(pd_data: pd.DataFrame, target_variable, categorical_predictor_list):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    selected_predictors = []

    for predictor in categorical_predictor_list:
        category_group_lists = pd_data.groupby(predictor)[target_variable].apply(list)
        anova_results = f_oneway(*category_group_lists)

        # If the ANOVA P-Value is <0.05, that means we reject H0
        if anova_results[1] < 0.05:
            print(predictor, 'is correlated with', target_variable, '| P-Value:', anova_results[1])
            selected_predictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', target_variable, '| P-Value:', anova_results[1])

    return selected_predictors


def check_and_create_folder(folder_path, creation_info=True):
    try:
        os.makedirs(folder_path)
        print('output folder has been created: ' + folder_path)
    except:
        if creation_info:
            print(folder_path + ' Already exists -->> Content will be overwritten.')
