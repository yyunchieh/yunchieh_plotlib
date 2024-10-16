import shap
import matplotlib.pyplot as plt

def plot_shap(model, data, title="SHAP Summary Plot"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    for i in range(len(shap_values)):
        plt.title(f"{title} - Class {i+1}")
        shap.summary_plot(shap_values[i], data, show=False)
        plt.xticks(rotation=45)
        plt.show()