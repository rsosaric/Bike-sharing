import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import settings as setts


def save_py_fig_to_file(fig, output_name, plot_dpi=None):
    if plot_dpi is not None:
        fig.savefig(output_name, dpi=plot_dpi)
    else:
        fig.savefig(output_name)
    plt.close()


def plot_several_hists_from_data_frame(data_frame: pd.DataFrame, col_labels: list, output_file: str = ""):
    hist_plot = data_frame.hist(bins=15, column=col_labels, grid=False,
                                figsize=(18, 10), color='#86bf91')

    ratio_hist_ax = hist_plot[0][0]
    save_py_fig_to_file(ratio_hist_ax.get_figure(), output_file, plot_dpi=setts.plot_dpi)


def plot_categorical_data_distribution(data_frame: pd.DataFrame, col_label: str,
                                       output_file: str = ""):
    sns.set_theme(style="ticks", color_codes=True)
    fig = sns.catplot(x=col_label, kind="count", palette="ch:.25", data=data_frame)

    save_py_fig_to_file(fig, output_file, plot_dpi=setts.plot_dpi)


def plot_categorical_data_vs_reference_variable(data_frame: pd.DataFrame, col_label: str, reference_variable: str,
                                                plot_type: str, output_file: str = ""):
    sns.set_theme(style="ticks", color_codes=True)

    if plot_type == "points":
        fig = sns.catplot(x=col_label, y=reference_variable, data=data_frame)
    elif plot_type == "box":
        fig = sns.catplot(x=col_label, y=reference_variable, data=data_frame, kind="box")
    else:
        raise AssertionError("plot mode not implemented")
    save_py_fig_to_file(fig, output_file, plot_dpi=setts.plot_dpi)


def plot_continuous_data_vs_reference_variable(data_frame: pd.DataFrame, col_label: str, reference_variable: str,
                                               plot_type: str, output_file: str = ""):
    if plot_type == "2d_map":
        fig = sns.displot(data_frame, x=col_label, y=reference_variable, cbar=True, cbar_kws={"drawedges": False})
    elif plot_type == "kde":
        fig = sns.displot(data_frame, x=col_label, y=reference_variable, kind="kde", fill=True)
    elif plot_type == "scatter":
        fig = sns.scatterplot(data=data_frame, x=col_label, y=reference_variable).get_figure()
    else:
        raise AssertionError("plot mode not implemented")

    save_py_fig_to_file(fig, output_file, plot_dpi=setts.plot_dpi)

