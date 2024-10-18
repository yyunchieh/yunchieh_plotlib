#Histogram Function
import matplotlib.pyplot as plt

def plot_histogram(ax, data, column, bins=30, **kwargs):
    ax.hist(data[column], 
            bins=bins, 
            color = "skyblue", 
            edgecolor = "white", 
            alpha=0.7)
    
    if 'set_title' in kwargs.keys():
        ax.set_title(kwargs['set_title'])
    
    if 'set_xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['set_xlabel'])
         
    if 'set_ylabel' in kwargs:
        ax.set_ylabel(kwargs['set_ylabel'])

    if 'set_xlim' in kwargs.keys():
        ax.set_xlim(**kwargs['set_xlim'])
        
    if 'set_xticks' in kwargs.keys():
        ax.set_xticks(**kwargs['set_xticks'])

    if 'tick_params' in kwargs.keys():
        ax.tick_params(**kwargs['tick_params'])

   
    ax.grid(axis="y", alpha=0.3)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")