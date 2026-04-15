import pandas as pd

from house_price.config import OUTPUT_DIR


def load_data(data_path) -> pd.DataFrame:
    """Doc du lieu goc va bo cot Id vi khong phuc vu du doan."""
    df = pd.read_csv(data_path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def save_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Thong ke so luong va ty le missing cua tung cot."""
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
    """Luu tuong quan voi SalePrice va thong ke gia theo Neighborhood."""
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
