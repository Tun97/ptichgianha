# 🏠 House Price Prediction

## 📌 Giới thiệu
Dự án House Price Prediction nhằm xây dựng mô hình dự đoán giá nhà dựa trên các đặc trưng như diện tích, số phòng, vị trí và các yếu tố liên quan khác.  
Đây là một bài toán hồi quy (Regression) phổ biến trong lĩnh vực Machine Learning và có ứng dụng thực tiễn cao trong bất động sản.

## 🎯 Mục tiêu
- Xử lý và làm sạch dữ liệu (Data Cleaning)
- Phân tích mối quan hệ giữa giá nhà và các yếu tố
- Trực quan hóa dữ liệu (Visualization)
- Xây dựng mô hình dự đoán giá nhà
- Đánh giá hiệu quả mô hình (MAE, RMSE)

## 📂 Cấu trúc thư mục
house-price-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│
├── requirements.txt
├── README.md
└── report.docx

## 📊 Dataset
- Nguồn: Kaggle (House Prices Dataset)
- Số lượng mẫu: ~1460
- Các thuộc tính chính:
  - LotArea
  - OverallQual
  - YearBuilt
  - GrLivArea
  - Bedroom
  - SalePrice (target)

## ⚙️ Quy trình thực hiện

### 1. Data Preprocessing
- Xử lý dữ liệu thiếu
- Encode dữ liệu
- Chuẩn hóa dữ liệu

### 2. EDA
- Scatter plot
- Heatmap correlation

### 3. Modeling
- Linear Regression
- Train/Test split (80/20)

### 4. Evaluation
- MAE
- RMSE

## 🚀 Cài đặt
pip install -r requirements.txt

## ▶️ Chạy chương trình
python src/train_model.py

## 👥 Thành viên nhóm
- Tăng Anh Tuấn
- Nguyễn Quang Linh
- Vương Tuấn Hưng
- Đặng Đức Trí
