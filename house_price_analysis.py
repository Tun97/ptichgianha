from house_price.config import DATA_PATH
from house_price.data_processing import (
    load_data,
    save_missing_summary,
    save_relationship_analysis,
)
from house_price.modeling import train_and_evaluate
from house_price.reporting import build_summary
from house_price.visualization import (
    create_correlation_heatmap,
    create_scatter_plots,
    ensure_output_dirs,
)


def main() -> None:
    """Chay toan bo quy trinh phan tich gia nha tu du lieu goc den ket qua cuoi."""
    ensure_output_dirs()
    df = load_data(DATA_PATH)

    # Buoc 1: thong ke missing va phan tich moi quan he voi gia nha.
    missing_summary = save_missing_summary(df)
    top_correlations, neighborhood_prices = save_relationship_analysis(df)

    # Buoc 2: tao cac bieu do phuc vu bao cao.
    create_scatter_plots(df)
    create_correlation_heatmap(df)

    # Buoc 3: huan luyen mo hinh va danh gia tren tap test.
    metrics = train_and_evaluate(df)

    # Buoc 4: luu ban tom tat ket qua de de trich vao README/bao cao.
    build_summary(df, missing_summary, top_correlations, neighborhood_prices, metrics)

    print("Da hoan thanh phan tich va huan luyen mo hinh.")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print("Cac tep ket qua da duoc luu trong thu muc outputs/.")


if __name__ == "__main__":
    main()
