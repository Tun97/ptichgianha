import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from house_price.config import FIGURES_DIR


def ensure_output_dirs() -> None:
    """Tao thu muc de luu bang ket qua va hinh ve."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def create_scatter_plots(df: pd.DataFrame) -> None:
    """Ve scatter plot giua SalePrice va cac bien co y nghia thuc tien cao."""
    features = ["GrLivArea", "TotalBsmtSF", "GarageArea", "OverallQual"]
    plot_df = df[features + ["SalePrice"]].dropna().copy()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        sns.scatterplot(
            data=plot_df,
            x=feature,
            y="SalePrice",
            alpha=0.65,
            s=40,
            ax=ax,
            color="#1f77b4",
        )
        ax.set_title(f"SalePrice vs {feature}")
        ax.ticklabel_format(style="plain", axis="y")

    fig.suptitle("Scatter Plot Giua Gia Nha Va Cac Yeu To Chinh", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scatter_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_correlation_heatmap(df: pd.DataFrame) -> None:
    """Ve heatmap cho nhom bien so hoc tuong quan manh nhat voi SalePrice."""
    numeric_df = df.select_dtypes(include="number")
    top_features = (
        numeric_df.corr()["SalePrice"]
        .abs()
        .sort_values(ascending=False)
        .head(11)
        .index
    )
    corr_matrix = numeric_df[top_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
    plt.title("Heatmap Tuong Quan Cac Bien Quan Trong")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
