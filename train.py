"""
================================================================================
 주제: 청소년 스마트폰 의존도/사용시간 · 수면의 질에 따른 구강건강 예측 ML 파이프라인
 데이터: KYRBS 2020 (복합표본설계 가중치 W 전면 적용)
 목표: 의료/헬스케어용 — Recall ≈ 0.85 목표, XGBoost 기반 최종 모델
 분할: 6:2:2 (Train : Validation : Test)
 최적화: Optuna TPE (FAST_TRIALS: 30) + 5-Fold Stratified CV
 임계값: Youden's Index 기반 (Streamlit 서비스용)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning) # LightGBM 등 이름 관련 경고 무시

import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone

# ── 라이브러리 임포트 (선택적) ──────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    fbeta_score
)
import joblib

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[!] optuna 없음 → pip install optuna")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost 없음 → pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[!] lightgbm 없음 → pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[!] catboost 없음 → pip install catboost")

# =============================================================================
# ── 0. 전역 설정 (경로 구조 최적화) ──────────────────────────────────────────
# =============================================================================
DATA_PATH      = "./data/processed/kyrbs2020_clean_v1.csv" # 데이터 경로
MODELS_DIR     = "./models" # 학습된 모델, 스케일러, 메타데이터 저장 경로
PLOTS_DIR      = "./plots"  # 시각화 이미지 저장 경로

WEIGHT_COL     = "W" 
RANDOM_SEED    = 42
TEST_SIZE      = 0.20      # 전체의 20% → Test
VAL_RATIO      = 0.25      # 나머지 80% 중 25% → Val (전체의 20% 달성, 최종 6:2:2)
CV_FOLDS       = 5
OPTUNA_TRIALS  = 30        

# 폴더가 없으면 자동 생성
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = "Malgun Gothic"   # Windows
except Exception:
    pass

# =============================================================================
# ── 1. 데이터 로드 및 전처리
# =============================================================================
print("=" * 70)
print(" STEP 1 │ 데이터 로드 및 전처리(완료)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

if WEIGHT_COL not in df.columns:
    print(f"  [경고] 가중치 컬럼 '{WEIGHT_COL}' 없음 → 1.0으로 임시 설정")
    df[WEIGHT_COL] = 1.0

df["oral_poor"] = (
    df["oral_health"].astype(str).str.strip().str.lower() == "yes"
).astype(int)

ENCODE_MAP = {
    "gender"               : {"Female": 0, "Male": 1},
    "school"               : {"Middle school": 0, "High school": 1},
    "grade"                : {"Low": 0, "Middle": 1, "High": 2},
    "income"               : {"Low": 0, "Middle": 1, "High": 2},
    "anxiety"              : {"No": 0, "Mild": 1, "Moderate": 2, "Severe": 3},
    "stress"               : {"Low": 0, "Middle": 1, "High": 2},
    "despair"              : {"No": 0, "Yes": 1},
    "suicidal_thoughts"    : {"No": 0, "Yes": 1},
    "smartphone_use_day"   : {"≤3": 0, "3 ~ 5": 1, "5 ~ 8": 2, "≥8": 3},
    "smartphone_use_weekend": {"≤3": 0, "3 ~ 5": 1, "5 ~ 8": 2, "≥8": 3},
    "smartphone_dependence": {"No": 0, "Risk": 1},
    "sleep_quality"        : {"No": 0, "Yes": 1},
}
for col, mp in ENCODE_MAP.items():
    if col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].map(mp)

DEMO_VARS   = ["gender", "school", "grade", "income"]
MENTAL_VARS = ["anxiety", "stress", "despair", "suicidal_thoughts"]
SP_VAR      = ["smartphone_use_day", "smartphone_use_weekend", "smartphone_dependence"]
SL_VAR      = ["sleep_quality"]

ALL_FEATS = DEMO_VARS + MENTAL_VARS + SP_VAR + SL_VAR
TARGET    = "oral_poor"

VAR_KOR = {
    "gender": "성별", "school": "학교급", "grade": "학업성적",
    "income": "가구소득", "anxiety": "불안수준", "stress": "스트레스",
    "despair": "절망감", "suicidal_thoughts": "자살생각",
    "smartphone_use_day"    : "★주중 스마트폰 사용시간",
    "smartphone_use_weekend": "★주말 스마트폰 사용시간",
    "smartphone_dependence" : "★스마트폰 의존도(위험=1)",
    "sleep_quality"         : "★수면의 질(나쁨=1)",
}

df_m  = df[ALL_FEATS + [TARGET, WEIGHT_COL]].dropna().reset_index(drop=True)
X_raw = df_m[ALL_FEATS].to_numpy(dtype=float)
y_raw = df_m[TARGET].to_numpy(dtype=float)
w_raw = df_m[WEIGHT_COL].to_numpy(dtype=float)

w_final = w_raw

# ── 6:2:2 데이터 분할 ──────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(" STEP 2 │ 머신러닝 데이터 분할 (6:2:2 = Train:Val:Test)")
print("=" * 70)

X_tv, X_test, y_tv, y_test, w_tv, w_test = train_test_split(
    X_raw, y_raw, w_final, test_size=TEST_SIZE, stratify=y_raw, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X_tv, y_tv, w_tv, test_size=VAL_RATIO, stratify=y_tv, random_state=RANDOM_SEED
)

print(f"  Train  : {len(y_train):>6,}개 (60%)")
print(f"  Val    : {len(y_val):>6,}개 (20%)")
print(f"  Test   : {len(y_test):>6,}개 (20%)")

# =============================================================================
# ── 3. 스케일링 (원본 가중치 전면 적용) ───────────────────────────────────────────
# =============================================================================
class WeightedStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, sample_weight=None):
        if sample_weight is None:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
        else:
            self.mean_ = np.average(X, axis=0, weights=sample_weight)
            variance = np.average((X - self.mean_)**2, axis=0, weights=sample_weight)
            self.scale_ = np.sqrt(variance)
            self.scale_[self.scale_ == 0] = 1.0 
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, sample_weight=None):
        return self.fit(X, sample_weight).transform(X)

scaler = WeightedStandardScaler()
X_tr = scaler.fit_transform(X_train, sample_weight=w_train) 
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
y_tr, w_tr = y_train, w_train


#=============================================================================
# ── 4. 헬퍼 함수 ──────────────────────────────────────────────────────────────
#=============================================================================
def evaluate_model(name, model, X_te, y_te, w_te, thresh, X_tr=None, y_tr=None, w_tr=None):
    prob_te = model.predict_proba(X_te)[:, 1]
    pred_te = (prob_te >= thresh).astype(int)

    auc_te = roc_auc_score(y_te, prob_te, sample_weight=w_te)

    row = {
        "Model"    : name,
        "Opt_Thresh": round(thresh, 3),
        "ROC-AUC"  : round(float(auc_te), 4),
        "Accuracy" : round(float(accuracy_score(y_te, pred_te, sample_weight=w_te)), 4),
        "Precision": round(float(precision_score(y_te, pred_te, sample_weight=w_te, zero_division=0)), 4),
        "Recall"   : round(float(recall_score(y_te, pred_te, sample_weight=w_te, zero_division=0)), 4),
        "F1-Score" : round(float(f1_score(y_te, pred_te, sample_weight=w_te, zero_division=0)), 4),
    }
    
    if X_tr is not None:
        prob_tr = model.predict_proba(X_tr)[:, 1]
        auc_tr  = roc_auc_score(y_tr, prob_tr, sample_weight=w_tr) 
        row["Train AUC"]   = round(auc_tr, 4)
        row["Overfit Gap"] = round(auc_tr - auc_te, 4)

    return row, prob_te

def cv_evaluate_multi(model, X, y, w, n_splits=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    metrics = {"AUC": [], "Recall": [], "Precision": [], "F2": []}
    
    for tr_i, val_i in skf.split(X, y):
        m = clone(model)
        if hasattr(m, "early_stopping_rounds") and getattr(m, "early_stopping_rounds") is not None:
            m.set_params(early_stopping_rounds=None)
            
        try:
            m.fit(X[tr_i], y[tr_i], sample_weight=w[tr_i])
        except TypeError:
            m.fit(X[tr_i], y[tr_i])
        
        if hasattr(m, "predict_proba"):
            prob_tr = m.predict_proba(X[tr_i])[:, 1]
            fpr_tr, tpr_tr, thresh_tr = roc_curve(y[tr_i], prob_tr, sample_weight=w[tr_i])
            target_idx = np.argmin(np.abs(tpr_tr - TARGET_RECALL))
            opt_thresh_cv = thresh_tr[target_idx]
            
            prob_val = m.predict_proba(X[val_i])[:, 1]
            pred_val = (prob_val >= opt_thresh_cv).astype(int)
            auc_val = roc_auc_score(y[val_i], prob_val, sample_weight=w[val_i])
        else:
            pred_val = m.predict(X[val_i])
            auc_val = 0 
            
        metrics["AUC"].append(auc_val)
        metrics["Recall"].append(recall_score(y[val_i], pred_val, sample_weight=w[val_i], zero_division=0))
        metrics["Precision"].append(precision_score(y[val_i], pred_val, sample_weight=w[val_i], zero_division=0))
        metrics["F2"].append(fbeta_score(y[val_i], pred_val, beta=2, sample_weight=w[val_i], zero_division=0))
        
    return {k: np.mean(v) for k, v in metrics.items()}

TARGET_RECALL = 0.85

def optuna_fast_objective(model_fn):
    m = model_fn()
    try:
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    except TypeError:
        m.fit(X_tr, y_tr)
        
    if hasattr(m, "predict_proba"):
        yprob_val = m.predict_proba(X_val_s)[:, 1]
        fpr, tpr, thresholds_roc = roc_curve(y_val, yprob_val, sample_weight=w_val)
        valid_mask = tpr >= TARGET_RECALL
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(thresholds_roc[valid_indices])]
        else:
            best_idx = np.argmin(np.abs(tpr - TARGET_RECALL))
        
        opt_thresh = thresholds_roc[best_idx]
        pred_val = (yprob_val >= opt_thresh).astype(int)
        return float(f1_score(y_val, pred_val, sample_weight=w_val, zero_division=0))
    else:
        pred = m.predict(X_val_s)
        return float(f1_score(y_val, pred, sample_weight=w_val, zero_division=0))

# =============================================================================
# ── 5. Optuna 하이퍼파라미터 최적화
# =============================================================================
print("\n" + "=" * 70)
print(f" STEP 4 │ Optuna 하이퍼파라미터 최적화  (Recall 극대화 탐색)")
print("=" * 70)

best_params: dict = {}
FAST_TRIALS = OPTUNA_TRIALS 

if HAS_OPTUNA:
    def lr_objective(trial):
        C   = trial.suggest_float("C", 1e-5, 10.0, log=True) 
        pen = trial.suggest_categorical("penalty", ["l1", "l2"])
        kw:dict= dict(C=C, penalty=pen, solver="saga", max_iter=1000, random_state=RANDOM_SEED)
        return optuna_fast_objective(lambda: LogisticRegression(**kw))
    study_lr = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study_lr.optimize(lr_objective, n_trials=FAST_TRIALS, show_progress_bar=True)
    best_params["LR"] = study_lr.best_params
    print(f"  [LR ]  best={study_lr.best_value:.4f}")

if HAS_OPTUNA:
    def rf_objective(trial):
        kw = dict(
            n_estimators = trial.suggest_int("n_estimators", 100, 250, step=50), 
            max_depth = trial.suggest_int("max_depth", 3, 10), 
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50), 
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"]), 
            random_state=RANDOM_SEED, n_jobs=-1
        )
        return optuna_fast_objective(lambda: RandomForestClassifier(**kw))
    study_rf = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study_rf.optimize(rf_objective, n_trials=FAST_TRIALS, show_progress_bar=True)
    best_params["RF"] = study_rf.best_params
    print(f"  [RF ]  best={study_rf.best_value:.4f}")

if HAS_OPTUNA and HAS_XGB:
    def xgb_objective(trial):
        kw = dict(
            n_estimators = trial.suggest_int("n_estimators", 100, 250, step=50), 
            max_depth = trial.suggest_int("max_depth", 3, 8), 
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.2, log=True), 
            subsample = trial.suggest_float("subsample", 0.6, 1.0), 
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0), 
            min_child_weight = trial.suggest_int("min_child_weight", 10, 3000), 
            eval_metric="logloss", 
            random_state=RANDOM_SEED, 
            n_jobs=-1
        )
        return optuna_fast_objective(lambda: XGBClassifier(**kw))
    study_xgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study_xgb.optimize(xgb_objective, n_trials=FAST_TRIALS, show_progress_bar=True)
    best_params["XGB"] = study_xgb.best_params
    print(f"  [XGB]  best={study_xgb.best_value:.4f}")

if HAS_OPTUNA and HAS_LGB:
    def lgb_objective(trial):
        kw = dict(
            n_estimators = trial.suggest_int("n_estimators", 100, 250, step=50), 
            max_depth = trial.suggest_int("max_depth", 3, 8), 
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.2, log=True), 
            num_leaves = trial.suggest_int("num_leaves", 15, 63), 
            subsample = trial.suggest_float("subsample", 0.6, 1.0), 
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0), 
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
        )
        return optuna_fast_objective(lambda: LGBMClassifier(**kw))
    study_lgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study_lgb.optimize(lgb_objective, n_trials=FAST_TRIALS, show_progress_bar=True)
    best_params["LGB"] = study_lgb.best_params
    print(f"  [LGB]  best={study_lgb.best_value:.4f}")

if HAS_OPTUNA and HAS_CAT:
    def cat_objective(trial):
        kw = dict(
            iterations = trial.suggest_int("iterations", 100, 250, step=50), 
            depth = trial.suggest_int("depth", 3, 8), 
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.2, log=True), 
            random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False
        )
        return optuna_fast_objective(lambda: CatBoostClassifier(**kw))
    study_cat = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study_cat.optimize(cat_objective, n_trials=FAST_TRIALS, show_progress_bar=True)
    best_params["CAT"] = study_cat.best_params
    print(f"  [CAT]  best={study_cat.best_value:.4f}")

# JSON 파라미터는 models 폴더에 저장
with open(f"{MODELS_DIR}/best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)

# =============================================================================
# ── 6. 최적 파라미터로 기본 모델 학습 (조기 종료 적용) ──────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(" STEP 5 │ 최종 머신러닝 모델 학습 (조기 종료 Early Stopping 적용)")
print("=" * 70)

def _p(key, default=None)->dict: 
    return best_params.get(key, {}) if best_params.get(key) else (default or {})

p_lr = _p("LR")
_pen = p_lr.get("penalty", "l2")
_lr_kw = dict(C=p_lr.get("C", 1.0), penalty=_pen, solver="saga", max_iter=2000, random_state=RANDOM_SEED)
if _pen == "elasticnet": 
    _lr_kw["l1_ratio"] = p_lr.get("l1_ratio", 0.5)

base_definitions = {
    "Logistic Regression": LogisticRegression(**_lr_kw),
    "Random Forest": RandomForestClassifier(**_p("RF"), random_state=RANDOM_SEED, n_jobs=-1),
}

if HAS_XGB: 
    base_definitions["XGBoost"] = XGBClassifier(
        **_p("XGB"), 
        eval_metric="logloss",
        early_stopping_rounds=30, 
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )

if HAS_LGB: 
    base_definitions["LightGBM"] = LGBMClassifier(
        **_p("LGB"), 
        random_state=RANDOM_SEED, 
        n_jobs=-1, 
        verbose=-1
    )

if HAS_CAT: 
    base_definitions["CatBoost"] = CatBoostClassifier(
        **_p("CAT"), 
        random_seed=RANDOM_SEED, 
        verbose=0, 
        allow_writing_files=False
    )

trained_models: dict = {}
for name, model in base_definitions.items():
    print(f"  ■ {name:20s} 학습 중...", end=" ", flush=True)
    t0 = time.time()
    try:
        if name == "LightGBM":
            import lightgbm as lgb
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val_s, y_val)], eval_sample_weight=[w_val], callbacks=[lgb.early_stopping(30, verbose=False)])
        elif name == "XGBoost":
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val_s, y_val)], sample_weight_eval_set=[w_val], verbose=False)
        elif name == "CatBoost":
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val_s, y_val), early_stopping_rounds=30, verbose=False)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr)
    except TypeError:
        model.fit(X_tr, y_tr)
    print(f"완료 ({time.time()-t0:.1f}s)")
    trained_models[name] = model

print("\n" + "=" * 70)
print(f" STEP 6 │ {CV_FOLDS}-Fold CV (Precision, F2, AUC, Recall 확인)")
print("=" * 70)
cv_results: dict = {}

for name, model in trained_models.items():
    res = cv_evaluate_multi(model, X_tr, y_tr, w_tr, n_splits=CV_FOLDS)
    cv_results[name] = res
    print(f" ■ {name:20s}")
    print(f"   ↳ AUC: {res['AUC']:.4f} | Recall: {res['Recall']:.4f} | Precision: {res['Precision']:.4f} | F2-Score: {res['F2']:.4f}")

# =============================================================================
# ── 8. 최종 테스트셋 성능 평가 (동적 임계값 적용) ─────────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(f" STEP 7 │ Test셋 최종 성능 (Recall ≈ {TARGET_RECALL} 목표 임계값 적용)")
print("=" * 70)

result_rows: list = []
probs_dict:  dict = {}

for name, model in trained_models.items():
    yprob_val = model.predict_proba(X_val_s)[:, 1]
    fpr, tpr, thresholds_roc = roc_curve(y_val, yprob_val, sample_weight=w_val)
    
    valid_mask = tpr >= TARGET_RECALL
    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(thresholds_roc[valid_indices])]
    else:
        best_idx = np.argmin(np.abs(tpr - TARGET_RECALL))
    opt_thresh = thresholds_roc[best_idx]
            
    row, prob_te = evaluate_model(
        name, model,
        X_te=X_test_s, y_te=y_test, w_te=w_test,
        thresh=opt_thresh, 
        X_tr=X_tr,     y_tr=y_tr,   w_tr=w_tr
    )
    result_rows.append(row)
    probs_dict[name] = prob_te

res_df = pd.DataFrame(result_rows).set_index("Model")

display_cols = ["Opt_Thresh", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
print(f"\n  ┌─ 성능 비교표 (Test, Recall 내림차순 정렬) " + "─" * 20)
print(res_df[display_cols].sort_values(by=["Recall", "ROC-AUC"], ascending=[False, False]).round(4).to_string())

if "Overfit Gap" in res_df.columns:
    print("\n  ┌─ 과적합 진단 (Train AUC vs Test AUC) " + "─" * 30)
    for mname in res_df.index:
        g = res_df.loc[mname, "Overfit Gap"]
        if pd.notna(g):
            flag = "⚠️  과적합 주의" if float(g) > 0.05 else "✅ 정상"
            print(f"    {mname:25s}  Gap={g:.4f}  {flag}")

best_model_name = res_df["Recall"].idxmax()
best_model_obj  = trained_models[best_model_name]
best_prob_te    = probs_dict[best_model_name]
best_thresh     = res_df.loc[best_model_name, "Opt_Thresh"]
best_pred_te    = (best_prob_te >= best_thresh).astype(int)

print(f"\n  🏆 최고 성능 모델 (Recall≈{TARGET_RECALL}, AUC 기준): {best_model_name}")
print(f"     Opt_Thresh={res_df.loc[best_model_name,'Opt_Thresh']}  "
      f"Accuracy={res_df.loc[best_model_name,'Accuracy']}  "
      f"AUC={res_df.loc[best_model_name,'ROC-AUC']}  "
      f"Recall={res_df.loc[best_model_name,'Recall']}")

# CSV 결과물은 models 폴더에 저장
res_df.to_csv(f"{MODELS_DIR}/ml_performance_table_recall.csv", encoding="utf-8-sig")

# =============================================================================
# ── 9. 시각화 (plots 폴더에 저장) ─────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(" STEP 8 │ 시각화 저장")
print("=" * 70)

COLORS = ["#2980b9","#27ae60","#e74c3c","#f39c12","#8e44ad","#1abc9c","#e67e22","#2c3e50","#c0392b"]

# ── 10-1. ROC 커브
fig, ax = plt.subplots(figsize=(9, 7))
for (name, prob), col in zip(probs_dict.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, prob, sample_weight=w_test)
    auc = roc_auc_score(y_test, prob, sample_weight=w_test)
    lw  = 2.5 if name == best_model_name else 1.5
    ls  = "-"  if name == best_model_name else "--"
    ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", color=col, linewidth=lw, linestyle=ls)
ax.plot([0,1],[0,1], "k--", alpha=0.4, linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve 비교", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_roc_curves.png", dpi=150)
plt.close()

# ── 10-2. 최고 모델 Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best_pred_te, sample_weight=w_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm.astype(int), display_labels=["구강 양호(0)", "구강 불량(1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_model_name}\n(threshold={best_thresh:.3f})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_confusion_matrix_best.png", dpi=150)
plt.close()

# ── 10-3. Feature Importance
fi_target = None
for fi_name in ["CatBoost", "XGBoost", "LightGBM", "Random Forest"]:
    if fi_name in trained_models and hasattr(trained_models[fi_name], "feature_importances_"):
        fi_target = (fi_name, trained_models[fi_name])
        break

if fi_target:
    fi_name, fi_model = fi_target
    fi_vals = fi_model.feature_importances_
    fi_df = pd.DataFrame({"feature": ALL_FEATS, "importance": fi_vals, "label": [VAR_KOR.get(f, f) for f in ALL_FEATS]}).sort_values("importance", ascending=True)
    bar_cols = ["#e74c3c" if "★" in lbl else "#95a5a6" for lbl in fi_df["label"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(fi_df["label"], fi_df["importance"], color=bar_cols, edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"변수 중요도 ({fi_name})\n★ = 핵심 변수 (스마트폰·수면)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_feature_importance.png", dpi=150)
    plt.close()

# =============================================================================
# ── 10. XGBoost Youden's Index 계산 및 모델 저장 ────────────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(" STEP 9 │ XGBoost Youden's Index 계산 + 모델 저장")
print("=" * 70)

xgb_model = trained_models["XGBoost"]
yprob_val_xgb  = xgb_model.predict_proba(X_val_s)[:, 1]
fpr_y, tpr_y, thresholds_y = roc_curve(y_val, yprob_val_xgb, sample_weight=w_val)

j_scores = tpr_y - fpr_y 
youden_best_idx    = np.argmax(j_scores)
youden_threshold   = float(thresholds_y[youden_best_idx])
youden_j_value     = float(j_scores[youden_best_idx])
youden_sensitivity = float(tpr_y[youden_best_idx])
youden_specificity = float(1 - fpr_y[youden_best_idx])

print(f"\n  ┌─ XGBoost Youden's Index 결과 ─────────────────────")
print(f"  │ Youden's J        = {youden_j_value:.4f}")
print(f"  │ 최적 임계값       = {youden_threshold:.4f}")
print(f"  │ 민감도(Recall)    = {youden_sensitivity:.4f}")
print(f"  │ 특이도(Specificity)= {youden_specificity:.4f}")
print(f"  └─────────────────────────────────────────────────────")

# ── 모델 및 메타데이터는 models 폴더에 저장 ──
joblib.dump(xgb_model, f"{MODELS_DIR}/xgboost_model.pkl")
joblib.dump(scaler,    f"{MODELS_DIR}/scaler.pkl")

meta = {
    "model_name"        : "XGBoost",
    "features"          : ALL_FEATS,
    "target"            : TARGET,
    "youden_threshold"  : youden_threshold,
    "youden_j"          : youden_j_value,
    "youden_sensitivity": youden_sensitivity,
    "youden_specificity": youden_specificity,
    "target_recall"     : TARGET_RECALL,
    "recall_threshold"  : float(best_thresh) if best_model_name == "XGBoost" else float(res_df.loc["XGBoost", "Opt_Thresh"]),
    "encode_map"        : {k: {str(kk): vv for kk, vv in v.items()} for k, v in ENCODE_MAP.items()},
}
with open(f"{MODELS_DIR}/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ 저장 완료:")
print(f"     - {MODELS_DIR}/xgboost_model.pkl")
print(f"     - {MODELS_DIR}/scaler.pkl")
print(f"     - {MODELS_DIR}/model_meta.json")
print(f"     - {MODELS_DIR}/best_params.json")
print(f"     - {MODELS_DIR}/ml_performance_table_recall.csv")
print(f"     - {PLOTS_DIR}/01_roc_curves.png 외 2건")

# =============================================================================
# ── 11. 최종 요약 출력 ────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 70)
print(" 최종 요약 ─ 구강건강 분석 통합 파이프라인 (ML 예측)")
print("=" * 70)

final_display_df = res_df[["Opt_Thresh", "Accuracy", "ROC-AUC", "Precision", "Recall", "F1-Score"]].copy()
final_display_df.columns = ["임계값", "정확도(Acc)", "AUC", "정밀도(Pre)", "재현율(Rec)", "F1-Score"]

final_display_df.index = [f"🏆 {idx}" if idx == best_model_name else f"   {idx}" for idx in final_display_df.index]

print(f"\n  📈 머신러닝 성능 비교 (Test셋, 재현율(Recall) 최우선 내림차순)")
print("  " + "─" * 75)
print(final_display_df.sort_values(by=["재현율(Rec)", "AUC"], ascending=[False, False]).round(4).to_string())
print("  " + "─" * 75)

print(f"\n  [핵심 결과 요약]")
print(f"  목표 재현율(Recall) ≈ {TARGET_RECALL} 에서 AUC가 가장 높은 모델은 '{best_model_name}' 입니다.")
print(f"  ▶ 임계값 {best_thresh:.3f} 적용 시 Recall={res_df.loc[best_model_name,'Recall']:.4f}, "
      f"Precision={res_df.loc[best_model_name,'Precision']:.4f}, "
      f"AUC={res_df.loc[best_model_name,'ROC-AUC']:.4f}")
print(f"\n  [Streamlit 서비스용 — XGBoost Youden's Index 기준]")
print(f"  ▶ Youden's 최적 임계값 = {youden_threshold:.4f}")
print(f"    민감도(Sensitivity) = {youden_sensitivity:.4f}, "
      f"특이도(Specificity) = {youden_specificity:.4f}, "
      f"J = {youden_j_value:.4f}\n")