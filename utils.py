# utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Correlation Matrix")
    return fig