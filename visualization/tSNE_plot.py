from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


#t-SNE function

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