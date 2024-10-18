import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_umap(data, labels, title="UMAP Plot", cmap = "set2", random_state=42):
    reducer = umap.UMAP(random_state=random_state)
    X_umap = reducer.fit_transform(data)
    
    colors = cm.get_cmap(cmap, len(set(labels)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for category in set(labels):
        idx = labels == category
        ax.scatter(X_umap[idx, 0], X_umap[idx, 1], color=colors(category), label=f"class {category + 1}")

    ax.legend(title="Classes")
    ax.set_title(title)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    plt.show()