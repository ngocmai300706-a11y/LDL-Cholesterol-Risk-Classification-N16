import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =====================================================================
# HÀM XỬ LÝ DÙNG CHUNG CHO ABLATION STUDY
# =====================================================================
RANDOM_STATE = 42
MAX_MISSING_RATIO = 0.4
CORR_THRESHOLD = 0.85
K_BEST = 15


class CorrelationFilter(BaseEstimator, TransformerMixin):

  def __init__(self, threshold=0.85):
    self.threshold = threshold

  def fit(self, X, y=None):
    X = np.asarray(X, dtype=float)
    corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
    upper = np.triu(corr, k=1)
    drop = [
        j for j in range(upper.shape[1]) if (upper[:, j] > self.threshold).any()
    ]
    self.keep_ = np.array([j for j in range(X.shape[1]) if j not in drop])
    return self

  def transform(self, X):
    return np.asarray(X, dtype=float)[:, self.keep_]


def run_experiment(use_lipid, use_biopro, model_name):
  # 1. Đọc và gộp dữ liệu
  df_demo = pd.read_sas('DEMO_L.xpt')
  df_bmx = pd.read_sas('BMX_L.xpt')
  df_trigly = pd.read_sas('TRIGLY_L.xpt')
  df_tchol = pd.read_sas('TCHOL_L.xpt')
  df_hdl = pd.read_sas('HDL_L.xpt')
  df_smq = pd.read_sas('SMQ_L.xpt')
  df_paq = pd.read_sas('PAQ_L.xpt')
  df_slq = pd.read_sas('SLQ_L.xpt')
  df_biopro = pd.read_sas('BIOPRO_L.xpt')
  df_bpxo = pd.read_sas('BPXO_L.xpt')

  merged_full = (
      df_demo.merge(df_bmx, on='SEQN', how='inner')
      .merge(df_trigly, on='SEQN', how='inner')
      .merge(df_tchol, on='SEQN', how='inner')
      .merge(df_hdl, on='SEQN', how='inner')
      .merge(df_bpxo, on='SEQN', how='left')
      .merge(df_smq, on='SEQN', how='left')
      .merge(df_paq, on='SEQN', how='left')
      .merge(df_slq, on='SEQN', how='left')
      .merge(df_biopro, on='SEQN', how='left')
  )

  # 2. Làm sạch cơ bản
  df_clean = merged_full.dropna(subset=['LBDLDL']).copy()
  if 'RIDAGEYR' in df_clean.columns:
    df_clean = df_clean[df_clean['RIDAGEYR'] >= 18]
  if 'BMXBMI' in df_clean.columns:
    df_clean = df_clean[(df_clean['BMXBMI'] >= 10) & (df_clean['BMXBMI'] <= 90)]

  for col in df_clean.columns:
    if col.startswith(('PAQ', 'SMQ', 'SLQ')):
      df_clean[col] = df_clean[col].replace([7, 9], np.nan)
    elif col.startswith(('PAD', 'SMD')):
      df_clean[col] = df_clean[col].replace([777, 999, 7777, 9999], np.nan)

  df_clean['risk_level'] = pd.cut(
      df_clean['LBDLDL'],
      bins=[-np.inf, 100, 160, np.inf],
      right=False,
      labels=[0, 1, 2],
  ).astype(int)

  drop_cols = ['SEQN', 'risk_level', 'SDDSRVYR', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA']
  for col in df_clean.columns:
    if col.startswith('LBDLDL') or col.startswith('WT') or col.endswith('SI'):
      drop_cols.append(col)
    elif col.endswith('LC') or (
        col.startswith(('BMI', 'BMD')) and col != 'BMXBMI'
    ):
      drop_cols.append(col)

  if not use_biopro:
    drop_cols += [c for c in df_biopro.columns if c != 'SEQN']

  if not use_lipid:
    lipid_prefixes = (
        'LBXTC',
        'LBDTC',
        'LBDHDD',
        'LBXTLG',
        'LBDTLG',
        'LBXSCH',
        'LBDSCH',
        'LBXSTR',
        'LBDSTR',
    )
    drop_cols += [c for c in df_clean.columns if c.startswith(lipid_prefixes)]

  drop_cols = [c for c in set(drop_cols) if c in df_clean.columns]
  X_raw = df_clean.drop(columns=drop_cols).select_dtypes(include=[np.number])
  y = df_clean['risk_level']

  if 'RIDRETH3' in X_raw.columns:
    X_raw = pd.get_dummies(X_raw, columns=['RIDRETH3'], prefix='RACE', dtype=float)
  X_raw = X_raw.drop(columns=['RIDRETH1'], errors='ignore')

  if 'LBXTLG' in X_raw.columns:
    X_raw['LBXTLG_log'] = np.log1p(X_raw['LBXTLG'])

  # 3. Chia tập dữ liệu
  X_train, X_test, y_train, y_test = train_test_split(
      X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
  )
  missing_ratio = X_train.isna().mean()
  keep_cols = missing_ratio[missing_ratio < MAX_MISSING_RATIO].index
  X_train, X_test = X_train[keep_cols], X_test[keep_cols]

  k_best = min(K_BEST, X_train.shape[1])

  if model_name == 'Logistic Regression':
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
  elif model_name == 'Random Forest':
    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
  else:
    clf = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')

  pipe = ImbPipeline([
      ('scaler', StandardScaler()),
      ('imputer', KNNImputer(n_neighbors=5)),
      ('var', VarianceThreshold(threshold=0.0)),
      ('corr', CorrelationFilter(threshold=CORR_THRESHOLD)),
      ('kbest', SelectKBest(score_func=f_classif, k=k_best)),
      ('smote', SMOTE(random_state=RANDOM_STATE)),
      ('clf', clf),
  ])

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  return accuracy_score(y_test, y_pred), f1_score(
      y_test, y_pred, average='macro'
  )


# =====================================================================
# CHẠY THÍ NGHIỆM ABLATION
# =====================================================================
if __name__ == '__main__':
  print('=== BẮT ĐẦU CHẠY NGHIÊN CỨU CẮT BỎ (ABLATION STUDY) ===')

  scenarios = [
      {
          'name': 'Kịch bản 1: Đầy đủ (Invasive + Lipid + Biopro)',
          'lipid': True,
          'biopro': True,
      },
      {
          'name': 'Kịch bản 2: Không xâm lấn (Bỏ Lipid & Biopro)',
          'lipid': False,
          'biopro': False,
      },
  ]

  ablation_results = []
  for scen in scenarios:
    print(f"\nĐang chạy: {scen['name']}...")
    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
      acc, f1m = run_experiment(scen['lipid'], scen['biopro'], model_name)
      ablation_results.append({
          'Scenario': scen['name'],
          'Model': model_name,
          'Accuracy': acc,
          'Macro-F1': f1m,
      })

  df_ablation = pd.DataFrame(ablation_results)
  print(
      '\n================ KẾT QUẢ ABLATION STUDY ================\n'
  )
  print(df_ablation.round(4).to_string(index=False))
