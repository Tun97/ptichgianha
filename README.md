# House Price Prediction

## 1. Giới thiệu
Dự án này thực hiện bài toán dự đoán giá nhà từ bộ dữ liệu `train.csv` của Kaggle House Prices. Đây là bài toán hồi quy với biến mục tiêu là `SalePrice`. Mục tiêu của nhóm là xử lý dữ liệu thiếu, phân tích mối quan hệ giữa giá nhà và các yếu tố ảnh hưởng, trực quan hóa dữ liệu bằng scatter plot và xây dựng mô hình hồi quy để dự đoán giá.

## 2. Mục tiêu bài toán
- Xử lý dữ liệu thiếu trong tập huấn luyện.
- Phân tích mối quan hệ giữa `SalePrice` và các biến quan trọng.
- Trực quan hóa dữ liệu bằng scatter plot và heatmap tương quan.
- Xây dựng mô hình hồi quy dự đoán giá nhà.
- Đánh giá mô hình bằng MAE và RMSE.

## 3. Dataset sử dụng
- File chính: `train.csv`
- File kiểm tra bổ sung: `test.csv`
- Số dòng dữ liệu trong `train.csv`: 1460
- Số cột ban đầu: 81
- Biến mục tiêu: `SalePrice`

Một số đặc trưng có tương quan mạnh với giá nhà:
- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `YearBuilt`

## 4. Quy trình thực hiện
### Bước 1. Đọc và kiểm tra dữ liệu
Dữ liệu được đọc từ `train.csv`. Cột `Id` không mang nhiều ý nghĩa cho dự đoán nên được loại bỏ khỏi quá trình mô hình hóa.

### Bước 2. Xử lý dữ liệu thiếu
Trong project hiện có hai hướng xử lý:

1. Script `clean.py`
Script này làm sạch dữ liệu theo hướng thủ công:
- Điền `"None"` cho các cột mà giá trị thiếu mang nghĩa là không có, ví dụ `Alley`, `PoolQC`, `Fence`, `GarageType`.
- Điền `0` cho một số cột số học mà giá trị thiếu có thể hiểu là không có diện tích hoặc không có hạng mục đó, ví dụ `GarageArea`, `TotalBsmtSF`.
- Điền trung vị theo `Neighborhood` cho `LotFrontage`.
- Điền mode cho một số biến phân loại còn thiếu ít giá trị.
- Áp dụng ordinal encoding cho các biến có thứ bậc.
- One-hot encoding cho phần còn lại.
- Lưu dữ liệu sạch ra `train_cleaned.csv`.

2. Cấu trúc tách module với `house_price_analysis.py`
`house_price_analysis.py` là file chạy chính, còn phần logic được tách ra các module riêng trong thư mục `house_price/`:
- `config.py`: khai báo đường dẫn dữ liệu và thư mục output.
- `data_processing.py`: đọc dữ liệu, bỏ cột `Id`, thống kê missing và phân tích tương quan.
- `visualization.py`: tạo scatter plot và heatmap.
- `modeling.py`: xây dựng pipeline tiền xử lý và mô hình hồi quy.
- `reporting.py`: sinh file tóm tắt kết quả `analysis_summary.md`.

Pipeline này xử lý dữ liệu thiếu ngay trong lúc huấn luyện:
- Biến số học được điền bằng `median`.
- Biến phân loại được điền bằng `most_frequent`.
- Sau đó dữ liệu phân loại được one-hot encoding bằng `OneHotEncoder`.

Hướng thứ hai gọn hơn, dễ tái sử dụng hơn và được dùng để tạo kết quả cuối cùng của bài.

### Bước 3. Phân tích mối quan hệ giữa giá nhà và các yếu tố
Project thực hiện hai dạng phân tích chính:
- Tính hệ số tương quan giữa `SalePrice` và các biến số học.
- Thống kê giá trung bình và trung vị theo khu vực `Neighborhood`.

Kết quả nổi bật:
- `OverallQual` có tương quan mạnh nhất với `SalePrice`, khoảng `0.7910`.
- `GrLivArea` có tương quan khoảng `0.7086`.
- Các khu vực có giá trung bình cao gồm `NoRidge`, `NridgHt`, `StoneBr`.

### Bước 4. Trực quan hóa
Script `house_price_analysis.py` sinh ra:
- `scatter_plots.png`: gồm 4 scatter plot giữa `SalePrice` và các biến `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `OverallQual`.
- `correlation_heatmap.png`: heatmap tương quan giữa các biến số học quan trọng nhất.

### Bước 5. Xây dựng mô hình
Mô hình được sử dụng là `LinearRegression`.

Pipeline huấn luyện gồm:
- Chia dữ liệu train/test theo tỷ lệ `80/20`.
- Xử lý dữ liệu thiếu bằng `SimpleImputer`.
- Mã hóa biến phân loại bằng `OneHotEncoder`.
- Huấn luyện hồi quy tuyến tính trên `log1p(SalePrice)`.
- Chuyển ngược kết quả dự đoán về đơn vị giá thật bằng `expm1`.

Việc biến đổi log cho biến mục tiêu giúp giảm ảnh hưởng của phân phối lệch phải và cải thiện sai số so với hồi quy trực tiếp trên giá gốc.

### Bước 6. Đánh giá mô hình
Kết quả hiện tại trên tập test:
- Train size: `1168`
- Test size: `292`
- MAE: `16970.2`
- RMSE: `25418.78`

Ý nghĩa:
- MAE cho biết sai lệch tuyệt đối trung bình giữa giá dự đoán và giá thực tế vào khoảng 16.97 nghìn USD.
- RMSE lớn hơn MAE vì nhạy với các điểm dự đoán sai nhiều.

## 5. Giải thích chi tiết các file trong project

### File dữ liệu
- `train.csv`: bộ dữ liệu huấn luyện chính, chứa cả đặc trưng đầu vào và cột mục tiêu `SalePrice`.
- `test.csv`: bộ dữ liệu test của Kaggle, không chứa `SalePrice`, có thể dùng để dự đoán khi cần nộp kết quả.
- `train_cleaned.csv`: dữ liệu sau khi được làm sạch bởi `clean.py`.
- `data_description.txt`: tài liệu mô tả ý nghĩa từng cột trong bộ Ames Housing, rất quan trọng khi xử lý giá trị thiếu đúng ngữ nghĩa.

### File mã nguồn
- `read.py`: file đọc thử `train.csv`, in kích thước dữ liệu và xem nhanh vài dòng đầu. Vai trò chính là kiểm tra dữ liệu ban đầu.
- `clean.py`: script tiền xử lý dữ liệu theo hướng thủ công, bao gồm điền thiếu, mã hóa ordinal, tạo thêm một số feature và lưu `train_cleaned.csv`.
- `house_price_analysis.py`: entrypoint của chương trình. File này gọi lần lượt các module xử lý, trực quan hóa, mô hình hóa và báo cáo.
- `house_price/config.py`: định nghĩa `DATA_PATH`, `OUTPUT_DIR`, `FIGURES_DIR`.
- `house_price/data_processing.py`: chứa các hàm đọc dữ liệu, thống kê missing và phân tích mối quan hệ với `SalePrice`.
- `house_price/visualization.py`: chứa các hàm vẽ scatter plot và heatmap.
- `house_price/modeling.py`: chứa hàm huấn luyện mô hình `LinearRegression` và tính MAE, RMSE.
- `house_price/reporting.py`: chứa hàm tổng hợp kết quả ra file markdown.
- `README.md`: tài liệu mô tả dự án và cách chạy.

### File kết quả trong thư mục `outputs/`
- `outputs/missing_summary.csv`: thống kê số lượng và tỷ lệ missing của các cột còn thiếu trong dữ liệu gốc.
- `outputs/top_correlations.csv`: danh sách các biến có tương quan mạnh nhất với `SalePrice`.
- `outputs/top_neighborhood_prices.csv`: thống kê top khu vực có giá nhà trung bình cao nhất.
- `outputs/model_metrics.csv`: lưu các chỉ số đánh giá mô hình như MAE và RMSE.
- `outputs/analysis_summary.md`: bản tóm tắt nhanh toàn bộ kết quả phân tích.
- `outputs/figures/scatter_plots.png`: hình scatter plot phục vụ báo cáo.
- `outputs/figures/correlation_heatmap.png`: hình heatmap tương quan phục vụ báo cáo.

### File tài liệu
- `Mô Tả bài toán _Phân Công công việc.docx`: tài liệu mô tả bài toán và phân chia công việc trong nhóm.
- `Chủ đề 4_Nhóm 3_Bài Thi.docx`: file báo cáo hoặc bài nộp của nhóm.

## 6. Cấu trúc thư mục thực tế
```text
Housse Prices/
|-- train.csv
|-- test.csv
|-- train_cleaned.csv
|-- data_description.txt
|-- read.py
|-- clean.py
|-- house_price_analysis.py
|-- house_price/
|   |-- __init__.py
|   |-- config.py
|   |-- data_processing.py
|   |-- visualization.py
|   |-- modeling.py
|   `-- reporting.py
|-- README.md
|-- outputs/
|   |-- analysis_summary.md
|   |-- missing_summary.csv
|   |-- model_metrics.csv
|   |-- top_correlations.csv
|   |-- top_neighborhood_prices.csv
|   `-- figures/
|       |-- scatter_plots.png
|       `-- correlation_heatmap.png
`-- *.docx
```

## 7. Cách chạy chương trình
### Cài đặt thư viện
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Chạy phân tích và huấn luyện mô hình
```bash
python house_price_analysis.py
```

Sau khi chạy xong, tất cả kết quả sẽ được lưu trong thư mục `outputs/`.

### Nếu muốn chạy bước làm sạch riêng
```bash
python clean.py
```

## 8. Kết luận
Project đã hoàn thành các yêu cầu chính của đề bài:
- Có xử lý dữ liệu thiếu.
- Có phân tích mối quan hệ giữa giá nhà và các yếu tố.
- Có trực quan hóa bằng scatter plot.
- Có xây dựng mô hình hồi quy dự đoán giá.
- Có đánh giá mô hình bằng MAE và RMSE.

Mô hình hiện tại là baseline tốt, Nếu cần cải thiện độ chính xác, có thể thêm các mô hình mạnh hơn như Ridge, Random Forest, XGBoost hoặc thực hiện feature engineering sâu hơn.

## 9. Thành viên nhóm
- Tăng Anh Tuấn - 20221964
- Nguyễn Quang Linh - 20221862
- Vương Tuấn Hưng - 20221845
- Đặng Đức Trí - 20221901
