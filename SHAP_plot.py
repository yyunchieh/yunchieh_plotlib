import shap
import matplotlib.pyplot as plt

def plot_shap(model, data, title="SHAP Summary Plot"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    for i in range(len(shap_values)):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"{title} - Class {i+1}")
        shap.summary_plot(shap_values[i], data, show=False)
        ax.set_xticks(rotation=45)
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        plt.show()