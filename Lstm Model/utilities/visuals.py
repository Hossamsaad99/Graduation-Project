import matplotlib.pyplot as plt


def plotting(data1, title, x_label="Date", fig_size=(10, 6), y_label=None, data2=None, legend_d1=None, legend_d2=None,
             save_plot=False, plot_name=None):
    """
    Plotting the given data into 10 X 6 figure

    Args:
        data1 - data to be plotted
        (str) title - the figure title
        (str) x_label - x axis label
        (str) y_label - y axis label (if given)
        data2 - data to be plotted (if given)
        (str) legend_d1 - legend specifies data1
        (str) legend_d2 - legend specifies data2
        (bool) save plot - if not False, will save the plot given specific name
        (str) plot_name - the desired name to save the plot with

    Returns:
        a plot with the specified args
    """

    plt.figure(figsize=fig_size)
    plt.plot(data1, color="blue")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if data2 is not None:
        plt.plot(data2, color="red")
        plt.legend([legend_d1, legend_d2], loc="upper left")
    if save_plot is not False:
        plt.savefig(plot_name)
    return plt.show()
