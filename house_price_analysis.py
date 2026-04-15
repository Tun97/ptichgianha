from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("train.csv")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def save_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_summary = (
        df.isnull()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_ratio=lambda frame: frame["missing_count"] / len(df))
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )
    missing_summary.to_csv(OUTPUT_DIR / "missing_summary.csv")
    return missing_summary


def save_relationship_analysis(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    numeric_corr = (
        df.corr(numeric_only=True)["SalePrice"]
        .drop("SalePrice")
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    top_correlations = numeric_corr.head(10)
    top_correlations.rename("correlation_with_saleprice").to_csv(
        OUTPUT_DIR / "top_correlations.csv",
        header=True,
    )

    neighborhood_prices = (
        df.groupby("Neighborhood", dropna=False)["SalePrice"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
        .head(10)
    )
    neighborhood_prices.to_csv(OUTPUT_DIR / "top_neighborhood_prices.csv")
    return top_correlations, neighborhood_prices


def create_scatter_plots(df: pd.DataFrame) -> None:
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


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, np.log1p(y_train))
    predictions = np.expm1(model.predict(X_test))

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model": "LinearRegression on log1p(SalePrice)",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
    }

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return metrics


def build_summary(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    top_correlations: pd.Series,
    neighborhood_prices: pd.DataFrame,
    metrics: dict,
) -> None:
    lines = [
        "# House Price Analysis Summary",
        "",
        f"- So dong du lieu: {len(df)}",
        f"- So cot dac trung + target: {df.shape[1]}",
        f"- So cot con thieu du lieu: {len(missing_summary)}",
        "",
        "## Top bien lien quan den SalePrice",
    ]

    for feature, score in top_correlations.items():
        lines.append(f"- {feature}: {score:.4f}")

    lines.extend(
        [
            "",
            "## Top khu vuc co gia trung binh cao",
        ]
    )

    for neighborhood, row in neighborhood_prices.iterrows():
        lines.append(
            f"- {neighborhood}: mean={row['mean']:.0f}, median={row['median']:.0f}, count={int(row['count'])}"
        )

    lines.extend(
        [
            "",
            "## Danh gia mo hinh",
            f"- Mo hinh: {metrics['model']}",
            f"- Train/Test split: {metrics['train_size']}/{metrics['test_size']}",
            f"- MAE: {metrics['mae']}",
            f"- RMSE: {metrics['rmse']}",
            "",
            "## Tep duoc tao",
            "- outputs/missing_summary.csv",
            "- outputs/top_correlations.csv",
            "- outputs/top_neighborhood_prices.csv",
            "- outputs/model_metrics.csv",
            "- outputs/figures/scatter_plots.png",
            "- outputs/figures/correlation_heatmap.png",
        ]
    )

    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    df = load_data()

    missing_summary = save_missing_summary(df)
    top_correlations, neighborhood_prices = save_relationship_analysis(df)
    create_scatter_plots(df)
    create_correlation_heatmap(df)
    metrics = train_and_evaluate(df)
    build_summary(df, missing_summary, top_correlations, neighborhood_prices, metrics)

    print("Da hoan thanh phan tich va huan luyen mo hinh.")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print("Cac tep ket qua da duoc luu trong thu muc outputs/.")


if __name__ == "__main__":
    main()
