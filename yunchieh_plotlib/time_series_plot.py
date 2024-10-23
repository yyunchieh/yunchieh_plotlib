import pandas as pd
import matplotlib.pyplot as plt

#Time Series Plot Function
def time_series_plot(ax, data, x_column, y_column, **kwargs):
    plt.sca(ax)
    
    ax.plot(data[x_column], data[y_column], color = "blue", linewidth=2)

    if 'set_title' in kwargs:
        ax.set_title(kwargs['set_title'])
    
    if 'set_xlabel' in kwargs:
        ax.set_xlabel(kwargs['set_xlabel'])

    if 'set_ylabel' in kwargs:
        ax.set_ylabel(kwargs['set_ylabel'])

    if 'set_xlim' in kwargs:
        ax.set_xlim(kwargs['set_xlim'])

    if 'set_ylim' in kwargs:
        ax.set_ylim(kwargs['set_ylim'])

    if 'set_xticks' in kwargs:
        ax.set_xticks(kwargs['set_xticks'])

    if 'tick_params' in kwargs:
        ax.tick_params(**kwargs['tick_params'])
    
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")