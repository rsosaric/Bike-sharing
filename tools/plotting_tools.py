import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def check_and_create_folder(folder_path, creation_info=True):
    try:
        os.makedirs(folder_path)
        print('output folder has been created: ' + folder_path)
    except:
        if creation_info:
            print(folder_path + ' Already exists -->> Content will be overwritten.')


def save_py_fig_to_file(fig, output_name, plot_dpi=None):
    if plot_dpi is not None:
        fig.savefig(output_name, dpi=plot_dpi)
    else:
        fig.savefig(output_name)
    plt.close(fig)


def plot_hist_from_data_frame(data_frame: pd.DataFrame, col_label: str, output_file: str = "",
                              title: str = "", xlabel: str = "", ylabel: str = ""):
    hist_plot = data_frame.hist(bins=20, column=col_label, grid=False,
                                figsize=(8, 8), sharex=True, color='#86bf91')
    ratio_hist_ax = hist_plot[0][0]
    ratio_hist_ax.set_title(title)
    ratio_hist_ax.set_xlabel(xlabel)
    ratio_hist_ax.set_ylabel(ylabel)
    #
    # if output_file != "":
    #
    # else:
    #     hist_plot.show()


def plot_several_hists_from_data_frame(data_frame: pd.DataFrame, col_labels: list, output_file: str = "",
                                       title: str = "", xlabel: str = "", ylabel: str = ""):
    hist_plot = data_frame.hist(bins=15, column=col_labels, grid=False,
                                figsize=(18, 10), color='#86bf91')

    ratio_hist_ax = hist_plot[0][0]
    save_py_fig_to_file(ratio_hist_ax.get_figure(), output_file)


