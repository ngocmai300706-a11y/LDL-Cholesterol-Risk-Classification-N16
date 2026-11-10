# -*- coding: utf-8 -*-
"""
THÍ NGHIỆM ABLATION STUDY & CHỨNG MINH LIPID CIRCULARITY (GAP 1)
Cho dự án: LDL Cholesterol Levels Classification (NHANES 2021-2023)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- CẤU HÌNH
RANDOM_STATE = 42
MAX_MISSING_RATIO = 0.4
N_SPLITS, N_REPEATS = 5, 3
CLASS_NAMES = ['Optimal', 'Near optimal', 'Borderline high', 'High', 'Very high']
BINS = [100, 130, 160, 190]  # ATP III Tiers

# ---------------------------------------------------------------- TIỀN XỬ LÝ DỮ LIỆU
def prepare_data_from_df(df):
    """
    Xử lý từ DataFrame đã merge (nhanes_full_merged_raw.csv hoặc tương tự)
    """
    df = df.dropna(subset=['LBDLDL']).copy()
    if 'RIDAGEYR' in df.columns:
        df = df[df['RIDAGEYR'] >= 18]
    if 'BMXBMI' in df.columns:
        df = df[(df['BMXBMI'] >= 10) & (df['BMXBMI'] <= 90)]

    # Xử lý mã thiếu NHANES
    for c in df.columns:
        if c.startswith(('PAQ', 'SMQ', 'SLQ')):
            df[c] = df[c].replace([7, 9], np.nan)
        elif c.startswith(('PAD', 'SMD')):
            df[c] = df[c].replace([777, 999, 7777, 9999], np.nan)

    # Tạo nhãn target 5 lớp ATP III
    y = pd.Series(np.digitize(df['LBDLDL'], BINS), index=df.index, name='ldl_class')

    # Loại bỏ các biến rò rỉ và biến thiết kế mẫu
    drop_cols = ['SEQN', 'SDDSRVYR', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'RIDEXMON', 'RIDRETH1']
    for c in df.columns:
        if c.startswith('LBDLDL') or c.startswith('WT'):
            drop_cols.append(c)
        elif c.endswith('LC') or (c.startswith(('BMI', 'BMD')) and c != 'BMXBMI'):
            drop_cols.append(c)
        elif c.endswith('SI'):
            drop_cols.append(c)

    drop_cols = [c for c in set(drop_cols) if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

    if 'RIDRETH3' in X.columns:
        X = pd.get_dummies(X, columns=['RIDRETH3'], prefix='RACE', dtype=float)

    # Phân lập nhóm biến Lipid và Phi-Lipid
    lipid_prefix = ('LBXTC', 'LBDTC', 'LBDHDD', 'LBXTLG', 'LBDTLG', 'LBXSCH', 'LBDSCH', 'LBXSTR', 'LBDSTR')
    lipid_present = [c for c in X.columns if c.startswith(lipid_prefix)]

    if not lipid_present:
        raise RuntimeError('Không tìm thấy các cột lipid (LBXTC, LBDHDD, LBXTLG) trong dữ liệu.')

    X_lipid = X[lipid_present].copy()
    X_non = X.drop(columns=lipid_present).copy()

    return X_lipid, X_non, X, y, df

# ---------------------------------------------------------------- PIPELINE
class MissingFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_ratio=0.4):
        self.max_ratio = max_ratio

    def fit(self, X, y=None):
        self.mask_ = np.isnan(np.asarray(X, float)).mean(axis=0) < self.max_ratio
        if not self.mask_.any():
            self.mask_[:] = True
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]

def make_pipe():
    return Pipeline([
        ('missing', MissingFilter(MAX_MISSING_RATIO)),
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
    ])

# ---------------------------------------------------------------- THÍ NGHIỆM E1, E2, E3
def run_experiments(df_raw):
    X_lipid, X_non, X_all, y, df = prepare_data_from_df(df_raw)
    print(f"Tổng số mẫu n = {len(y)} | Số đặc trưng: Lipid={X_lipid.shape[1]}, Phi-lipid={X_non.shape[1]}, Tất cả={X_all.shape[1]}")

    # E1. Kiểm tra công thức Friedewald
    print("\n" + "="*60)
    print("E1. KHẢO SÁT RÒ RỈ CÔNG THỨC FRIEDEWALD")
    print("="*60)
    if all(k in df.columns for k in ['LBXTC', 'LBDHDD', 'LBXTLG']):
        tc, hdl, tg = df['LBXTC'], df['LBDHDD'], df['LBXTLG']
        ldl_f = tc - hdl - tg / 5.0
        ok = ldl_f.notna() & df['LBDLDL'].notna()
        cls_f = np.digitize(ldl_f[ok], BINS)
        r2 = np.corrcoef(ldl_f[ok], df.loc[ok, 'LBDLDL'])[0, 1] ** 2
        print(f"R^2 (Friedewald vs LBDLDL): {r2:.4f}")
        print(f"Macro-F1 nếu dùng thuần công thức: {f1_score(y[ok], cls_f, average='macro'):.4f}")

    # E2. Thí nghiệm Ablation
    print("\n" + "="*60)
    print("E2. THÍ NGHIỆM ABLATION PHÂN TÁCH NHÓM BIẾN")
    print("="*60)
    sets = {'[A] Lipid-only': X_lipid, '[B] Non-lipid': X_non, '[C] All features': X_all}
    idx_tr, idx_te = train_test_split(y.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    results = []
    for name, X_data in sets.items():
        pipe = make_pipe().fit(X_data.loc[idx_tr], y.loc[idx_tr])
        pred = pipe.predict(X_data.loc[idx_te])
        yt = y.loc[idx_te]
        results.append({
            'Feature Set': name,
            'Test Acc': accuracy_score(yt, pred),
            'Test Macro-F1': f1_score(yt, pred, average='macro')
        })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

# ---------------------------------------------------------------- HÀM CHẠY CHÍNH
if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN TỆP DỮ LIỆU MERGED CỦA BẠN TẠI ĐÂY
    data_path = 'nhanes_full_merged_raw.csv' 
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        run_experiments(df_raw)
    else:
        print(f"Không tìm thấy file {data_path}. Vui lòng kiểm tra lại đường dẫn dữ liệu!")
