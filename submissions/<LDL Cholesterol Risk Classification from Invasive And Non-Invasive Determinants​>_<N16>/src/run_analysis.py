# Cài thêm nếu thiếu (Colab):  !pip install pytorch-tabnet lightgbm CODE của em mai (tiền xử lí) và của nanh ( học mô hình)
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold,
    f_classif,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print('Không có lightgbm -> bỏ qua LightGBM.')

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print('Không có pytorch-tabnet -> bỏ qua TabNet.')

warnings.filterwarnings('ignore')

# =====================================================================
# CẤU HÌNH
# =====================================================================
DATA_DIR = ''                 # thư mục chứa các file .xpt ('' = thư mục hiện tại; 'sample_data/' nếu để trong đó)

USE_LIPID_FEATURES = True     # True: giữ TC/HDL/TG (rò rỉ nhãn vì LDL tính từ chúng); False: bỏ để so sánh
USE_BIOPRO = False            # True: giữ nhóm sinh hóa BIOPRO

USE_VIF = True                # lọc đa cộng tuyến bằng VIF
USE_EMBEDDED = True           # chọn đặc trưng nhúng bằng độ quan trọng của Random Forest

RANDOM_STATE = 42
MAX_MISSING_RATIO = 0.4       # bỏ cột thiếu >= 40% (tính trong từng lần huấn luyện)
CORR_THRESHOLD = 0.85         # bỏ cột có |tương quan| > 0.85
VIF_THRESHOLD = 10.0          # bỏ cột có VIF > 10
K_BEST = 15                   # số đặc trưng giữ lại sau ANOVA F-test
EMBEDDED_THRESHOLD = '0.5*mean'   # giữ đặc trưng có độ quan trọng >= 0.5 x trung bình

CLASS_NAMES = ['Optimal', 'Near optimal', 'Borderline high', 'High', 'Very high']
TO_3_CLASS = np.array([0, 1, 1, 2, 2])   # gộp 5 lớp -> 3 lớp để so với các lần chạy trước

# Mô tả một số biến hay gặp (đối chiếu codebook NHANES trước khi trích dẫn)
FEATURE_DESC = {
    'LBXTC': 'Total cholesterol (mg/dL)', 'LBDHDD': 'HDL cholesterol (mg/dL)',
    'LBXTLG': 'Triglycerides (mg/dL)', 'LBXTLG_log': 'log(1 + triglycerides)',
    'RIDAGEYR': 'Age (years)', 'RIAGENDR': 'Sex', 'BMXBMI': 'Body mass index',
    'BMXWT': 'Weight', 'BMXHT': 'Height', 'BMXWAIST': 'Waist circumference',
    'BMXLEG': 'Upper leg length', 'BMXARML': 'Upper arm length',
    'BPXOSY1': 'Systolic BP (reading 1)', 'BPXODI1': 'Diastolic BP (reading 1)',
    'BPXOPLS1': 'Pulse (reading 1)', 'INDFMPIR': 'Family income-to-poverty ratio',
    'DMDEDUC2': 'Education level (adults)', 'DMDBORN4': 'Country of birth',
    'DMDMARTZ': 'Marital status', 'DMDHHSIZ': 'Household size',
    'DMQMILIZ': 'Served in US military', 'SLD012': 'Sleep hours (weekdays)',
    'PAD680': 'Sedentary minutes per day', 'SMQ020': 'Smoked >= 100 cigarettes in life',
}


# =====================================================================
# CÁC THÀNH PHẦN TÙY CHỈNH CỦA PIPELINE
# =====================================================================
class MissingColumnFilter(BaseEstimator, TransformerMixin):
    """Bỏ cột có tỷ lệ thiếu >= max_ratio (quyết định trên dữ liệu huấn luyện)."""

    def __init__(self, max_ratio=0.4):
        self.max_ratio = max_ratio

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mask_ = np.isnan(X).mean(axis=0) < self.max_ratio
        return self

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Bỏ cột có |tương quan| > threshold với một cột đứng trước nó."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        self.mask_ = np.ones(n, dtype=bool)
        if n >= 2:
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
            upper = np.triu(corr, k=1)
            for j in range(n):
                if (upper[:, j] > self.threshold).any():
                    self.mask_[j] = False
        return self

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class VIFFilter(BaseEstimator, TransformerMixin):
    """Lần lượt bỏ cột có VIF lớn nhất cho đến khi mọi VIF <= threshold.
    VIF_j = phần tử chéo thứ j của nghịch đảo ma trận tương quan."""

    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        keep = list(range(n))
        while len(keep) > 1:
            corr = np.nan_to_num(np.corrcoef(X[:, keep], rowvar=False))
            vif = np.diag(np.linalg.inv(corr + 1e-6 * np.eye(len(keep))))
            worst = int(np.argmax(vif))
            if vif[worst] <= self.threshold:
                break
            keep.pop(worst)
        self.mask_ = np.zeros(n, dtype=bool)
        self.mask_[keep] = True
        return self

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """ANOVA F-test, giữ k đặc trưng điểm cao nhất; tự giảm k nếu số cột < k."""

    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        scores, pvalues = f_classif(X, y)
        self.scores_ = np.nan_to_num(scores)
        self.pvalues_ = np.nan_to_num(pvalues, nan=1.0)
        k = min(self.k, X.shape[1])
        top = np.argsort(self.scores_)[::-1][:k]
        self.mask_ = np.zeros(X.shape[1], dtype=bool)
        self.mask_[top] = True
        return self

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper pytorch-tabnet theo chuẩn scikit-learn, có early stopping."""

    def __init__(self, max_epochs=40, batch_size=128, lr=2e-2, patience=10,
                 val_frac=0.1, random_state=42):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=self.val_frac, random_state=self.random_state, stratify=y)
        self.model_ = TabNetClassifier(
            verbose=0, seed=self.random_state, optimizer_params=dict(lr=self.lr))
        self.model_.fit(
            X_train=X_tr, y_train=y_tr, eval_set=[(X_va, y_va)],
            max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))


# =====================================================================
# BƯỚC 1: ĐỌC VÀ GỘP DỮ LIỆU
# =====================================================================
print('--- BƯỚC 1: ĐỌC VÀ GỘP CÁC FILE XPT ---')


def read(name):
    return pd.read_sas(os.path.join(DATA_DIR, name))


df_demo, df_bmx = read('DEMO_L.xpt'), read('BMX_L.xpt')
df_trigly, df_tchol, df_hdl = read('TRIGLY_L.xpt'), read('TCHOL_L.xpt'), read('HDL_L.xpt')
df_smq, df_paq, df_slq = read('SMQ_L.xpt'), read('PAQ_L.xpt'), read('SLQ_L.xpt')
df_biopro, df_bpxo = read('BIOPRO_L.xpt'), read('BPXO_L.xpt')

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
print(f'Kích thước sau khi gộp: {merged_full.shape[0]} dòng, {merged_full.shape[1]} cột')


# =====================================================================
# BƯỚC 2: LỌC, LÀM SẠCH, TẠO NHÃN, LOẠI ĐẶC TRƯNG THEO KIẾN THỨC MIỀN
# =====================================================================
print('\n--- BƯỚC 2: LỌC VÀ LÀM SẠCH ---')
df_clean = merged_full.dropna(subset=['LBDLDL']).copy()
df_clean = df_clean[df_clean['RIDAGEYR'] >= 18]
df_clean = df_clean[(df_clean['BMXBMI'] >= 10) & (df_clean['BMXBMI'] <= 90)]

for col in df_clean.columns:                       # mã "từ chối / không biết" -> NaN
    if col.startswith(('PAQ', 'SMQ', 'SLQ')):
        df_clean[col] = df_clean[col].replace([7, 9], np.nan)
    elif col.startswith(('PAD', 'SMD')):
        df_clean[col] = df_clean[col].replace([777, 999, 7777, 9999], np.nan)

# Nhãn 5 lớp theo ATP III: [<100), [100,130), [130,160), [160,190), [>=190)
y = pd.Series(np.digitize(df_clean['LBDLDL'], [100, 130, 160, 190]),
              index=df_clean.index, name='ldlc_level')

# Loại thủ công (domain-based filtering)
drop_cols = ['SEQN', 'SDDSRVYR', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA',
             'RIDEXMON']                          # RIDEXMON: mùa khám, biến hành chính
for col in df_clean.columns:
    if col.startswith('LBDLDL'):                    # mọi biến LDL -> rò rỉ nhãn
        drop_cols.append(col)
    elif col.startswith('WT'):                      # trọng số khảo sát
        drop_cols.append(col)
    elif col.endswith('LC') or (col.startswith(('BMI', 'BMD')) and col != 'BMXBMI'):
        drop_cols.append(col)                       # cờ trạng thái
    elif col.endswith('SI'):                        # bản đơn vị SI trùng lặp
        drop_cols.append(col)

if not USE_BIOPRO:
    drop_cols += [c for c in df_biopro.columns if c != 'SEQN']

if not USE_LIPID_FEATURES:
    lipid_prefixes = ('LBXTC', 'LBDTC', 'LBDHDD', 'LBXTLG', 'LBDTLG',
                      'LBXSCH', 'LBDSCH', 'LBXSTR', 'LBDSTR')
    drop_cols += [c for c in df_clean.columns if c.startswith(lipid_prefixes)]

drop_cols = [c for c in set(drop_cols) if c in df_clean.columns]
X_raw = df_clean.drop(columns=drop_cols).select_dtypes(include=[np.number])

if 'RIDRETH3' in X_raw.columns:                     # one-hot chủng tộc
    X_raw = pd.get_dummies(X_raw, columns=['RIDRETH3'], prefix='RACE', dtype=float)
X_raw = X_raw.drop(columns=['RIDRETH1'], errors='ignore')

if 'LBXTLG' in X_raw.columns:                       # log cho triglycerides
    X_raw['LBXTLG_log'] = np.log1p(X_raw['LBXTLG'])
    print('-> Đã áp dụng Log Transformation cho LBXTLG')

print(f'Kích thước sau khi xử lý: {X_raw.shape[0]} dòng, {X_raw.shape[1]} cột')
print('Phân bố nhãn:\n', y.value_counts().sort_index())


# =====================================================================
# BƯỚC 3: CHIA TRAIN / TEST
# =====================================================================
print('\n--- BƯỚC 3: CHIA TẬP TRAIN / TEST ---')
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f'Train: {len(y_train)} | Test: {len(y_test)}')
print('Số mẫu mỗi lớp trong train:', y_train.value_counts().sort_index().to_dict())

# k_neighbors của SMOTE phải nhỏ hơn số mẫu lớp nhỏ nhất trong mỗi fold huấn luyện
min_class_fold = int(y_train.value_counts().min() * 0.8)
SMOTE_K = int(max(1, min(5, min_class_fold - 1)))
print(f'SMOTE k_neighbors = {SMOTE_K}')


# =====================================================================
# BƯỚC 4: PIPELINE
# missing -> scale -> impute -> var -> corr -> (vif) -> kbest -> (embedded) -> SMOTE -> model
# =====================================================================
def selection_steps():
    steps = [
        ('missing', MissingColumnFilter(MAX_MISSING_RATIO)),
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('var', VarianceThreshold(threshold=0.0)),
        ('corr', CorrelationFilter(CORR_THRESHOLD)),
    ]
    if USE_VIF:
        steps.append(('vif', VIFFilter(VIF_THRESHOLD)))
    steps.append(('kbest', SafeSelectKBest(K_BEST)))
    if USE_EMBEDDED:
        steps.append(('embedded', SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            threshold=EMBEDDED_THRESHOLD)))
    return steps


def build_pipeline(clf):
    return ImbPipeline(selection_steps() + [
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)),
        ('clf', clf),
    ])


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                         early_stopping=True, random_state=RANDOM_STATE),
}
if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
if HAS_TABNET:
    models['TabNet'] = TabNetWrapper(max_epochs=40, batch_size=128, random_state=RANDOM_STATE)


# =====================================================================
# BƯỚC 5: BÁO CÁO CHỌN LỌC ĐẶC TRƯNG (ĐẶC TRƯNG NÀO ĐƯỢC CHỌN, VÌ SAO BỊ LOẠI)
# =====================================================================
STAGE_INFO = {
    'missing': ('Missing-ratio filter', f'thiếu >= {MAX_MISSING_RATIO:.0%} trong dữ liệu huấn luyện'),
    'var': ('VarianceThreshold', 'cột hằng số (phương sai = 0)'),
    'corr': ('Correlation filter', f'|r| > {CORR_THRESHOLD} với một đặc trưng đứng trước'),
    'vif': ('VIF filter', f'VIF > {VIF_THRESHOLD}'),
    'kbest': ('SelectKBest (ANOVA F)', f'không nằm trong top {K_BEST} điểm F'),
    'embedded': ('Embedded (RF importance)', f'độ quan trọng < {EMBEDDED_THRESHOLD}'),
}
SEL_KEYS = ['missing', 'var', 'corr', 'vif', 'kbest', 'embedded']


def trace_selection(pipe, input_names):
    """Đi qua từng bước lọc, ghi lại đặc trưng nào bị loại ở đâu."""
    current = np.array(input_names)
    dropped, log = [], []
    before_kbest = before_emb = None
    for key in SEL_KEYS:
        if key not in pipe.named_steps:
            continue
        mask = np.asarray(pipe.named_steps[key].get_support(), dtype=bool)
        if len(mask) != len(current):
            raise ValueError(f'Bước {key}: số cột không khớp ({len(mask)} vs {len(current)})')
        label, reason = STAGE_INFO[key]
        for f in current[~mask]:
            dropped.append({'feature': f, 'status': 'dropped', 'stage': label, 'reason': reason})
        log.append({'stage': label, 'features_in': len(current),
                    'features_out': int(mask.sum()), 'removed': int((~mask).sum())})
        if key == 'kbest':
            before_kbest = current.copy()
        if key == 'embedded':
            before_emb = current.copy()
        current = current[mask]
    return current, dropped, log, before_kbest, before_emb


def describe(f):
    if f.startswith('RACE_'):
        return 'Race/ethnicity indicator'
    return FEATURE_DESC.get(f, '')


def feature_selection_report(X, y):
    fs = Pipeline(selection_steps()).fit(X, y)
    final, dropped, log, b_k, b_e = trace_selection(fs, X.columns)

    fscore = pval = imp = {}
    if b_k is not None:
        kb = fs.named_steps['kbest']
        fscore, pval = dict(zip(b_k, kb.scores_)), dict(zip(b_k, kb.pvalues_))
    if b_e is not None:
        imp = dict(zip(b_e, fs.named_steps['embedded'].estimator_.feature_importances_))

    kept_reason = f'qua mọi bộ lọc: top {K_BEST} điểm F' + (
        f' và độ quan trọng RF >= {EMBEDDED_THRESHOLD}' if USE_EMBEDDED else '')
    kept = [{'feature': f, 'status': 'kept', 'stage': 'Selected', 'reason': kept_reason}
            for f in final]
    rep = pd.DataFrame(dropped + kept)
    rep['description'] = rep['feature'].map(describe)
    rep['F_score'] = rep['feature'].map(fscore)
    rep['p_value'] = rep['feature'].map(pval)
    rep['RF_importance'] = rep['feature'].map(imp)
    return rep, pd.DataFrame(log)


def selection_stability(X, y, n_splits=5):
    """Mỗi đặc trưng được chọn trong bao nhiêu fold (độ ổn định của việc chọn)."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    counts = {}
    for tr, _ in skf.split(X, y):
        p = Pipeline(selection_steps()).fit(X.iloc[tr], y.iloc[tr])
        final = trace_selection(p, X.columns)[0]
        for f in final:
            counts[f] = counts.get(f, 0) + 1
    return pd.Series(counts, name=f'số fold được chọn (/{n_splits})').sort_values(ascending=False)


print('\n=============================================================')
print('        BƯỚC 5: BÁO CÁO CHỌN LỌC ĐẶC TRƯNG')
print('=============================================================')
report, stage_log = feature_selection_report(X_train, y_train)
print('\nNhật ký loại bỏ theo từng bước:')
print(stage_log.to_string(index=False))

selected = report[report['status'] == 'kept'].copy()
sort_col = 'RF_importance' if USE_EMBEDDED else 'F_score'
selected = selected.sort_values(sort_col, ascending=False)
print(f'\nĐặc trưng được chọn ({len(selected)}):')
print(selected[['feature', 'description', 'F_score', 'p_value', 'RF_importance']]
      .round(4).to_string(index=False))

stab = selection_stability(X_train, y_train)
print('\nĐộ ổn định khi chọn (qua 5 fold):')
print(stab.to_string())

report.to_csv('feature_selection_report.csv', index=False)
stage_log.to_csv('feature_selection_log.csv', index=False)
stab.to_csv('feature_selection_stability.csv')

plot_df = selected.sort_values(sort_col)
plt.figure(figsize=(7, 0.38 * len(plot_df) + 1.5))
plt.barh(plot_df['feature'], plot_df[sort_col])
plt.xlabel(sort_col)
plt.title('Selected features')
plt.tight_layout()
plt.savefig('feature_selection_selected.png', dpi=150)
plt.show()


# =====================================================================
print('        BƯỚC 5: HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH')
print('=============================================================')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, clf in models.items():
    print(f'\n--- Mô hình: {name} ---')
    pipe = build_pipeline(clf)

    cv_f1_macro_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
    cv_f1_weighted_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)

    cv_f1_macro_mean = cv_f1_macro_scores.mean()
    cv_f1_weighted_mean = cv_f1_weighted_scores.mean()
    print(f'CV macro-F1 (5-fold): {cv_f1_macro_mean:.4f} ± {cv_f1_macro_scores.std():.4f}')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro_test = f1_score(y_test, y_pred, average='macro')
    f1_weighted_test = f1_score(y_test, y_pred, average='weighted')
    print(f'Test Accuracy: {acc:.4f} | Test macro-F1: {f1_macro_test:.4f}\n')

    print('Báo cáo phân loại (Classification Report):')
    print(classification_report(
        y_test, y_pred,
        target_names=['Optimal', 'Near Optimal', 'Borderline High', 'High', 'Very High'],
        zero_division=0,
    ))
    print('Ma trận nhầm lẫn (Confusion Matrix):')
    print(confusion_matrix(y_test, y_pred))

    results.append({
        'Model': name,
        'CV macro-F1': cv_f1_macro_mean,
        'CV weighted-F1': cv_f1_weighted_mean,
        'Test accuracy': acc,
        'Test macro-F1': f1_macro_test,
        'Test weighted-F1': f1_weighted_test,
    })
    if name == 'LightGBM':
        # Sửa lỗi IndexError bằng cách cập nhật tuần tự tên cột qua từng bộ lọc của Pipeline
        names = np.array(X_train.columns)
        names = names[pipe.named_steps['missing'].get_support()]
        names = names[pipe.named_steps['var'].get_support()]
        names = names[pipe.named_steps['corr'].get_support()]
        if USE_VIF:
            names = names[pipe.named_steps['vif'].get_support()]
        names = names[pipe.named_steps['kbest'].get_support()]
        if USE_EMBEDDED:
            names = names[pipe.named_steps['embedded'].get_support()]
        print(f'\nTop {len(names)} đặc trưng được chọn: {list(names)}')

print('\n=============================================================')
print('                      TỔNG KẾT')
print('=============================================================')
print(pd.DataFrame(results).round(4).to_string(index=False))
