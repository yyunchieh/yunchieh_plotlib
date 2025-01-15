#Histogram Function
import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.manifold import TSNE
import umap

# Histogram
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

# Shap Plot
def plot_shap(shap_values,ax, color_bar, **kwargs):
   
    plt.sca(ax)
    
    shap.plots.beeswarm(shap_values, show=False, s=24, plot_size=None, color_bar=False)
    
    if 'set_title' in kwargs.keys():
        ax.set_title(**kwargs['set_title'])
    
    if 'set_xlabel' in kwargs.keys():
        ax.set_xlabel(**kwargs['set_xlabel'])

    if 'set_xlim' in kwargs.keys():
        ax.set_xlim(**kwargs['set_xlim'])
        
    if 'set_xticks' in kwargs.keys():
        ax.set_xticks(**kwargs['set_xticks'])

    if 'tick_params' in kwargs.keys():
        ax.tick_params(**kwargs['tick_params'])
        #ax.tick_params(axis='both', which='major', labelsize=14, colors='black')

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    if color_bar == True:
       
        colorbar_ax = plt.colorbar(ax=ax, format='{x:.1f}')
        
        vmin, vmax = np.min(colorbar_ax.get_ticks()), np.max(colorbar_ax.get_ticks())
        colorbar_ax.mappable.set_clim(vmin=vmin, vmax=vmax)
        colorbar_ax.set_ticks(ticks = [vmin, vmax], labels=['Low', 'High'], size = 12)
        
        colorbar_ax.set_label(label='Feature value', size = 16, labelpad=20, y=0.5)
        
        colorbar_ax.outline.set_visible(False)
        colorbar_ax.ax.tick_params(axis='both', top=False, right=False, bottom=False, left=False)

#Time Series Plot 
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


#t-SNE Plot

def plot_tsne(X, y, ax, **kwargs):
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    #Data Type
    if is_numeric_dtype(y) and y.nunique() > 10:
        continuous = True

    else:
        continuous = False


    tsne = TSNE(n_components=2, random_state=0)
    embedding = tsne.fit_transform(X)


    plt.sca(ax)

    #color bar
    if continuous:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="viridis", s=kwargs.get("s",50), alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label("Continuous Value")

    #legend
    else:
        unique_classes = np.unique(y)
        for i, cls in enumerate(unique_classes):
            ax.scatter(embedding[y == cls, 0], embedding[y == cls, 1], label=f"Class{cls}", s=kwargs.get("s",50), alpha=0.7)
        ax.legend(title="Classes", loc="best", fontsize=10)

 
    if "set_title" in kwargs.keys():
        ax.set_title(kwargs["set_title"]["label"])

    if "set_xlabel" in kwargs.keys():
        ax.set_xlabel(kwargs["set_xlabel"])

    if "set_xlim" in kwargs.keys():
        ax.set_xlim(kwargs["set_xlim"])
        
    if "set_xticks" in kwargs.keys():
        ax.set_xticks(kwargs["set_xticks"])

    if "tick_params" in kwargs.keys():
        ax.tick_params(**kwargs["tick_params"])

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")


#UMAP Plot

def plot_umap(X, y, ax, **kwargs):

    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    #Data Type
    if is_numeric_dtype(y) and y.nunique() > 10 :
        continuous = True
    
    else:
        continuous = False

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    plt.sca(ax)

    #color bar
    if continuous:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', s=kwargs.get('s',50),alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label('Continuous value')

    #legend
    else:
        unique_classes = np.unique(y)
        for i, cls in enumerate(unique_classes):
            ax.scatter(embedding[y == cls, 0], embedding[y == cls, 1], label = f"Class{cls}", s=kwargs.get('s',50),alpha=0.7)
        ax.legend(title="Classes", loc="best", fontsize=10)


    if 'set_title' in kwargs.keys():
        ax.set_title(kwargs['set_title']["label"])
    
    if 'set_xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['set_xlabel'])

    if 'set_ylabel' in kwargs.keys():
        ax.set_label(kwargs['set_ylabel'])

    if 'set_xlim' in kwargs.keys():
        ax.set_xlim(kwargs['set_xlim'])
        
    if 'set_xticks' in kwargs.keys():
        ax.set_xticks(kwargs['set_xticks'])
        
    if 'tick_params' in kwargs.keys():
        ax.tick_params(**kwargs['tick_params'])

    
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")


# Box Plot
def plot_boxplot(ax, data, positions, box_props, **kwargs):
    bp = ax.boxplot(data, positions=positions, patch_artist= True)
    ax.spines[["right", "left", "top", "bottom"]].set_color("none")

    if "set_title" in kwargs:
        ax.set_title(**kwargs["set_title"])

    if "set_xlabel" in kwargs:
        ax.set_xlabel(**kwargs["set_xlabel"])

    if "set_ylabel" in kwargs:
        ax.set_ylabel(**kwargs["set_ylabel"])

    if "set_xticks" in kwargs:
        ax.set_xticks(**kwargs["set_xticks"])
    
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])

    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])

    if "set_ylim" in kwargs:
        ax.set_ylim(**kwargs["set_ylim"])

    if "set_yticks" in kwargs:
        ax.set_yticks(**kwargs["set_yticks"])

    if box_props != None:
        for item_name in box_props:
            for item, prop in zip(bp[item_name], box_props[item_name]):
                item.set(**prop)


def plot_errorbar(ax, x, y, yerr,fmt="o", errorbar_props=None, **kwargs):
    err_lines = ax.errorbar(x, y, yerr=yerr, fmt=fmt, **(errorbar_props or {}))
    if "set_title" in kwargs:
        ax.set_title(**kwargs["set_title"])

    if "set_xlabel" in kwargs:
        ax.set_xlabel(**kwargs["set_xlabel"])

    if "set_ylabel" in kwargs:
        ax.set_ylabel(**kwargs["set_ylabel"])

    if "set_xticks" in kwargs:
        ax.set_xticks(**kwargs["set_xticks"])
    
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])

    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])

    if "set_ylim" in kwargs:
        ax.set_ylim(**kwargs["set_ylim"])

    if "set_yticks" in kwargs:
        ax.set_yticks(**kwargs["set_yticks"])
    
    if "set_xlim" in kwargs:
        ax.set_xlim(**kwargs["set_xlim"])

    return err_lines