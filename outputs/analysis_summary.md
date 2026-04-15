# House Price Analysis Summary

- So dong du lieu: 1460
- So cot dac trung + target: 80
- So cot con thieu du lieu: 19

## Top bien lien quan den SalePrice
- OverallQual: 0.7910
- GrLivArea: 0.7086
- GarageCars: 0.6404
- GarageArea: 0.6234
- TotalBsmtSF: 0.6136
- 1stFlrSF: 0.6059
- FullBath: 0.5607
- TotRmsAbvGrd: 0.5337
- YearBuilt: 0.5229
- YearRemodAdd: 0.5071

## Top khu vuc co gia trung binh cao
- NoRidge: mean=335295, median=301500, count=41
- NridgHt: mean=316271, median=315000, count=77
- StoneBr: mean=310499, median=278000, count=25
- Timber: mean=242247, median=228475, count=38
- Veenker: mean=238773, median=218000, count=11
- Somerst: mean=225380, median=225500, count=86
- ClearCr: mean=212565, median=200250, count=28
- Crawfor: mean=210625, median=200624, count=51
- CollgCr: mean=197966, median=197200, count=150
- Blmngtn: mean=194871, median=191000, count=17

## Danh gia mo hinh
- Mo hinh: LinearRegression on log1p(SalePrice)
- Train/Test split: 1168/292
- MAE: 16970.2
- RMSE: 25418.78

## Tep duoc tao
- outputs/missing_summary.csv
- outputs/top_correlations.csv
- outputs/top_neighborhood_prices.csv
- outputs/model_metrics.csv
- outputs/figures/scatter_plots.png
- outputs/figures/correlation_heatmap.png