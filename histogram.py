import matplotlib.pyplot as plt

def plot_histogram(data, column, bins=30, title="Histogram", xlabel = "Values", ylabel="Frequency"):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(data[column], bins=bins, color = "skyblue", edgecolor = "white", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    plt.show()