import matplotlib.pyplot as plt

def plot_time_series(data, x_column, y_column, title = "Times Series Plot", xlabel = "Timestamp", ylabel = "Value", legend_label = "Value"):
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot(data[x_column], data[y_column], label = legend_label, color = "blue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(rotation=45)
    ax.grid(axis="y",alpha=0.3)
    ax.legend(loc="best")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    plt.show()