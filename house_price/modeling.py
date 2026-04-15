import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from house_price.config import OUTPUT_DIR


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Tien xu ly, huan luyen hoi quy tuyen tinh va tinh MAE, RMSE."""
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    # Bien so hoc duoc dien median de giam anh huong cua outlier.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    # Bien phan loai duoc dien mode va one-hot encode de dua vao mo hinh.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer cho phep xu ly khac nhau giua bien so hoc va bien phan loai.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Pipeline gom tien xu ly va mo hinh de tranh ro ri du lieu khi train/test split.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Log-transform target giup gia nha bot lech phai, thuong cho ket qua hoi quy tot hon.
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
