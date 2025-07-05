# -*- coding: utf-8 -*-
"""
Geophysical magnetic sensor dataset preprocessing + SMOTE oversampling + NPZ persistency
Steps:
1) Load class3.csv (621 features)
2) Stratified 70/30 split for train/test
3) Apply SMOTE(k=5) only on the training set to balance classes to [207, 207, 207]
4) Save to geoMagDataset.npz for downstream DataLoader
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    raise ImportError(
        "imbalanced-learn not found. Please install it via: pip install imbalanced-learn"
    ) from e


def get_dataset(
    csv_cls: str = "preprocessing/geoMagDataset/class3.csv",
    npz_path: str = "preprocessing/geoMagDataset/geoMagDataset.npz",
    test_ratio: float = 0.30,
    random_state: int = 42,
):
    # 1. Load data and shuffle ------------------------------------------------
    df = pd.read_csv(csv_cls, header=None).sample(frac=1, random_state=random_state)
    df = df[0].str.split(";", expand=True)         # split into 622 columns
    y = df[621]                                    # column 622 is the label
    X = df.drop(columns=[621]).astype(float)       # retain 0–620 => 621 features

    print("— Original data shape —")
    print(f"X: {X.shape} | y: {y.shape}")          # (376, 621)

    # 2. Min-Max scaling ------------------------------------------------------
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Stratified train/test split -----------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train X: {X_tr.shape} | Test X: {X_te.shape}")  # (263, 621)/(113, 621)

    # 4. Integer label encoding ----------------------------------------------
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr)
    y_te_int = le.transform(y_te)

    # 5. SMOTE oversampling (train set only) ----------------------------------
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr_int)

    print("SMOTE class distribution:", np.bincount(y_tr_res))  # [207 207 207]

    # 6. Save to NPZ ----------------------------------------------------------
    np.savez_compressed(
        npz_path,
        X_train=X_tr_res.astype(np.float32),
        y_train=y_tr_res.astype(np.int64),
        X_test=X_te.astype(np.float32),
        y_test=y_te_int.astype(np.int64),
        classes=le.classes_,
    )
    print(f"Data saved to {npz_path}")

    return X_tr_res, y_tr_res, X_te, y_te_int, le


if __name__ == "__main__":
    get_dataset()
