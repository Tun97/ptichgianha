import pandas as pd

from house_price.config import OUTPUT_DIR


def build_summary(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    top_correlations: pd.Series,
    neighborhood_prices: pd.DataFrame,
    metrics: dict,
) -> None:
    """Tong hop cac ket qua quan trong thanh file markdown de dua vao bao cao."""
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
