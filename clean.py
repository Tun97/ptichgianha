import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================
df = pd.read_csv("train.csv")

print("Kích thước ban đầu:", df.shape)
print("\nSố lượng missing trước khi clean:")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# =========================
# 2. Drop cột không cần thiết
# =========================
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)


# =========================
# 3. Các cột mà NA nghĩa là "không có"
# Theo data_description của Ames Housing
# =========================
none_cols = [
    'Alley',
    'MasVnrType',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature'
]

for col in none_cols:
    if col in df.columns:
        df[col] = df[col].fillna('None')


# =========================
# 4. Các cột số mà NA nghĩa là 0
# vì không có diện tích / không có garage / không có basement
# =========================
zero_cols = [
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'GarageCars',
    'GarageArea'
]

for col in zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)


# =========================
# 5. Cột năm xây garage
# Có thể fill = 0 hoặc median
# Ở đây dùng 0 để biểu diễn "không có garage"
# =========================
if 'GarageYrBlt' in df.columns:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)


# =========================
# 6. LotFrontage: fill median theo Neighborhood
# tốt hơn median toàn cột
# =========================
if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())


# =========================
# 7. Một số cột categorical còn thiếu ít giá trị
# fill theo mode
# =========================
mode_cols = [
    'MSZoning',
    'Utilities',
    'Exterior1st',
    'Exterior2nd',
    'Electrical',
    'KitchenQual',
    'SaleType',
    'Functional'
]

for col in mode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])


# =========================
# 8. Ordinal Encoding cho các cột có thứ bậc
# =========================
ordinal_mappings = {
    'ExterQual':     {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond':     {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual':      {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond':      {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC':     {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual':   {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu':   {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual':    {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond':    {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC':        {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},

    'BsmtExposure':  {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1':  {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2':  {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},

    'Functional':    {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},

    'Fence':         {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},

    'GarageFinish':  {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},

    'PavedDrive':    {'N': 0, 'P': 1, 'Y': 2},

    'LotShape':      {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},

    'LandSlope':     {'Sev': 1, 'Mod': 2, 'Gtl': 3},

    'CentralAir':    {'N': 0, 'Y': 1},

    'Street':        {'Grvl': 0, 'Pave': 1},

    'Alley':         {'None': 0, 'Grvl': 1, 'Pave': 2},

    'Utilities':     {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}
}

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)


# =========================
# 9. Feature Engineering cơ bản
# =========================
if set(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']).issubset(df.columns):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

if set(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']).issubset(df.columns):
    df['TotalBath'] = (
        df['FullBath'] +
        0.5 * df['HalfBath'] +
        df['BsmtFullBath'] +
        0.5 * df['BsmtHalfBath']
    )

if set(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']).issubset(df.columns):
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] +
        df['EnclosedPorch'] +
        df['3SsnPorch'] +
        df['ScreenPorch']
    )

if set(['YearBuilt', 'YearRemodAdd']).issubset(df.columns):
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']


# =========================
# 10. Log transform cho các cột numeric bị lệch mạnh
# Không bắt buộc, nhưng tốt cho regression
# =========================
skew_cols = [
    'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
]

for col in skew_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])


# =========================
# 11. One-hot encoding cho categorical còn lại
# =========================
df = pd.get_dummies(df, drop_first=True)


# =========================
# 12. Kiểm tra missing sau clean
# =========================
missing_after = df.isnull().sum().sum()
print("\nTổng missing sau khi clean:", missing_after)

if missing_after > 0:
    # phương án dự phòng: fill median cho numeric còn thiếu
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

print("Tổng missing cuối cùng:", df.isnull().sum().sum())
print("Kích thước sau clean:", df.shape)


# =========================
# 13. Lưu file sạch
# =========================
df.to_csv("train_cleaned.csv", index=False)
print("\nĐã lưu file: train_cleaned.csv")