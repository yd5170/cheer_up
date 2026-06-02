<div align="center">

# 🦷 청소년 스마트폰 의존·수면의 질에 따른<br>구강 건강 위험도 예측 서비스

**공공 보건 데이터 × 복합표본 통계 × XGBoost — 예방적 헬스케어 AI**

<br>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Final_Model-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-8B5CF6?style=for-the-badge)](https://shap.readthedocs.io)
[![Optuna](https://img.shields.io/badge/HPO-Optuna-00B2E2?style=for-the-badge)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

[📊 Streamlit Demo](https://github.com/yd5170/cheer_up.git) · [📁 데이터 출처](https://www.kdca.go.kr) · [📄 분석 보고서](file:///c:/Users/yd517/Desktop/model_performance_summary.txt)

</div>

---

## 📌 핵심 성과 지표

<div align="center">

| 지표 | 수치 | 비고 |
|:---:|:---:|:---:|
| 🎯 **Recall** | **85.21%** | 최우선 최적화 지표 (민감도) |
| 📈 **ROC-AUC** | **0.6483** | XGBoost 최종 모델 실측값 |
| ⚖️ **F1-Score** | **66.83%** | 정밀도 54.98% / 재현율 85.21% 조화 평균 |
| ⚙️ **임계값 방식** | **Youden's Index (0.398)** | 고정값 0.5 대신 위음성 방지 동적 컷오프 산출 |
| 🏆 **최종 모델** | **XGBoost** | AutoGluon 다중 비교 최적화 후 최종 선정 |

> 💡 헬스케어 도메인 특성상 **위음성(실제 위험 → 정상 판정) 최소화**를 목표로 Recall을 최우선 지표로 설정

</div>

---

## 🔍 프로젝트 개요

> **"스마트폰 과의존과 수면 부족이 청소년 구강 건강을 얼마나 위협하는가?"**

본 프로젝트는 제16차 청소년건강행태조사(KYRBS 2020) 공공데이터를 활용하여,  
청소년의 스마트폰 과의존·수면의 질이 구강 건강에 미치는 영향을 **통계 검정으로 증명**하고,  
머신러닝 기반 위험도 예측 결과를 **실서비스로 배포**한 End-to-End 헬스케어 프로젝트입니다.

### 문제 인식
| 현황 | 수치 |
|------|------|
| 청소년 스마트폰 이용률 | **97% 이상** |
| 주당 평균 사용 시간 | **12시간 이상** |
| 스마트폰 과의존 위험군 비율 | **약 40%** |

> 기존 서비스는 단순 사용 시간 규제·과의존 예방 수준에 머물러,  
> **예방 → 예측 → 의료기관 연계**를 하나의 흐름으로 제공하는 서비스가 부재했습니다.

---

## 📊 EDA 핵심 인사이트

> 복합표본 가중치 반영 + Rao-Scott 카이제곱 검정 기반 결과

```
📱 스마트폰 의존 위험군의 구강 증상 발생률 → 일반군 대비 약 1.43배 증가 확인 (p < 0.001)
😴 수면 부족 집단(수면 질 불량)에서 구강 질환 증상 비율이 53.89%로 증가 확인 (p < 0.001)
⚠️ 스마트폰 의존 + 수면 부족 이중 해당군 → 구강 증상 발생률 59.43%로 대조군(약 37.0%) 대비 1.61배 폭증
```

> 📌 **단순 상관이 아닌 복합표본 가중치를 반영한 통계 검정으로 모집단 수준에서 검증**

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| **언어** | Python 3.13, SQL |
| **데이터 분석** | Pandas, NumPy, Scikit-learn |
| **통계 분석** | SciPy (`chi2_contingency`) — Rao-Scott 카이제곱 검정 |
| **불균형 처리** | SMOTE, TomekLink |
| **모델링** | XGBoost ✅, CatBoost, LightGBM, TensorFlow, PyTorch |
| **AutoML** | AutoGluon (다중 모델 자동 비교) |
| **하이퍼파라미터 최적화** | Optuna (베이즈 최적화) |
| **XAI 해석** | SHAP, Integrated Gradients, Saliency |
| **시각화** | Matplotlib, Seaborn, Plotly |
| **웹 서비스** | Streamlit |
| **개발환경** | Jupyter Notebook, VS Code |
| **버전관리** | Git, GitHub |

---

## ⚙️ 분석 파이프라인

```
📥 데이터 수집          SAS 원시 파일(.sas7bdat) Python 직접 파싱
        │
        ▼
🔧 전처리               변수 추출 · 파생 변수 생성 · 결측치·이상치 처리
        │               [Pandas] [NumPy] [SQL]
        ▼
⚖️  변수 재구성          스마트폰 의존 이진화 / 구강 증상 4항목 → 이진 타겟
        │               복합표본 가중치(W) 정규화
        ▼
🔬 통계 검정            가중치 반영 교차분석 → Rao-Scott 카이제곱 검정
        │               [SciPy · chi2_contingency]
        ▼
📈 EDA 시각화           분포 · 상관관계 · 집단별 비교
        │               [Matplotlib] [Seaborn]
        ▼
🔀 불균형 처리          SMOTE(오버샘플링) + TomekLink(경계 노이즈 제거)
        │
        ▼
🤖 모델링               AutoGluon 다중 모델 자동 비교 → XGBoost 최종 선정
        │               Optuna 베이즈 최적화 · Youden's Index 동적 임계값
        ▼
🔍 XAI 해석             SHAP 변수 기여도 · Integrated Gradients · Saliency
        │
        ▼
📉 성능 검증            ROC 커브 · AUC · Recall 중심 평가 [Plotly 대시보드]
        │
        ▼
🌐 서비스 배포          Streamlit 실시간 예측 · 맞춤 가이드 · 치과 연계
```

---

## 🧠 핵심 기술 의사결정

> "왜 이 기술을 선택했는가" — 단순 사용이 아닌 판단 근거를 기록합니다

<details>
<summary><b>① 복합표본 가중치를 통계 검정 + 모델 학습 모두에 적용한 이유</b></summary>

**문제:** 층화집락추출 데이터는 관측값이 서로 독립이 아님  
→ 일반 카이제곱·일반 모델 학습 모두 편향 발생

**판단:**
- 통계 검정: Rao-Scott 보정으로 설계 효과(Design Effect) 반영
- 모델 학습: XGBoost `sample_weight` 파라미터에 가중치 직접 주입

**결과:** 전국 청소년 모집단을 실제로 대표하는 분석·예측 확보  
*(대부분의 프로젝트는 통계에만 가중치 반영 — 모델까지 이중 적용은 드문 케이스)*

</details>

<details>
<summary><b>② 일반 카이제곱 대신 Rao-Scott 검정을 선택한 이유</b></summary>

**문제:** 복합표본설계를 무시하면 1종 오류율 왜곡  
→ 독립 단순 표본 가정 위반

**판단:** 설계 효과(Design Effect)를 보정하는 Rao-Scott 검정 채택

**결과:** 스마트폰 의존·수면의 질 ↔ 구강 증상 간 연관성을 신뢰도 있게 검증

</details>

<details>
<summary><b>③ SMOTE에 TomekLink를 추가한 이유</b></summary>

**문제:** SMOTE만 사용 시 합성 샘플이 결정 경계 근처에 생성 → 경계 흐림

**판단:** TomekLink로 두 클래스 경계 노이즈 샘플 제거 (언더샘플링 조합)

**결과:** 결정 경계 선명화 → Recall·Precision 동시 안정화

</details>

<details>
<summary><b>④ Optuna + Youden's Index를 조합한 이유</b></summary>

**문제:**
- Grid Search는 탐색 공간 한계
- 고정 임계값 0.5는 헬스케어에 부적합 (위음성 = 실제 위험 → 정상 판정, 의료적으로 치명적)

**판단:**
- Optuna 베이즈 최적화로 하이퍼파라미터 자동 탐색
- Youden's Index: 민감도 + 특이도 합이 최대인 지점을 임계값으로 자동 산출

**결과:** Recall ≈ 0.85 확보 + 헬스케어 도메인 특성을 코드 수준에서 구현

</details>

<details>
<summary><b>⑤ SHAP·XAI를 적용한 이유</b></summary>

**문제:** 예측 결과만 출력하면 "왜 위험한가?" 설명 불가 → 의료 서비스 신뢰도 저하

**판단:**
- SHAP으로 변수별 기여도 시각화
- Integrated Gradients·Saliency로 딥러닝 예측 근거 해석

**결과:** 사용자에게 "어떤 요인이 위험도에 얼마나 영향을 줬는지" 설명 가능한 AI 구현

</details>

---

## 🎓 교과목 연계

> AI 의료서비스 모델 개발 과정 수료 내용이 프로젝트 전 단계에 적용되었습니다

| 교과목 | 프로젝트 적용 내용 |
|--------|----------------|
| 📘 디지털 헬스케어 프로그래밍<br>*(Python · 공공데이터 수집)* | SAS 원시 데이터 파싱 · 질병관리청 공공데이터 활용 · SQL 집계 |
| 📗 디지털 헬스케어 DB 구현<br>*(EDA · 기술통계 · 머신러닝)* | 결측치 처리 · 이진 분류화 · XGBoost·AutoGluon · Optuna |
| 📙 헬스케어 통계 분석<br>*(복합표본설계 · 교차분석)* | 복합표본 가중치 정규화 · Rao-Scott 카이제곱 검정 |
| 📕 세미프로젝트1<br>*(비정형 데이터 분석 시각화)* | 가중치 반영 집단별 비교 · Plotly 반응형 대시보드 |
| 📒 세미프로젝트2<br>*(바이오헬스케어 AI 비즈니스 모델링)* | SMOTE+TomekLink · SHAP · Integrated Gradients · Saliency |
| 📓 세미프로젝트3 / Final Project<br>*(헬스케어 분석모델 서비스 구현)* | Streamlit 실서비스 · 4유형 맞춤 가이드 · 치과 연계 |

---

## 📁 디렉토리 구조

```
Oral_Health_Prediction/
│
├── data/
│   ├── raw/                          # 원본 데이터 (KYRBS 2020 .sas7bdat)
│   ├── processed/                    # 전처리 완료 데이터 (.csv)
│   └── 논문, 원시자료지침서/           # 참고 문헌 및 지침서
│
├── models/
│   ├── xgboost_model.pkl             # ✅ 최종 선정 XGBoost 모델
│   ├── scaler.pkl                    # Weighted StandardScaler
│   ├── model_meta.json               # 변수 매핑 + Youden's Index 임계값
│   └── best_params.json              # Optuna 튜닝 결과
│
├── plots/
│   ├── 01_roc_curves.png             # 모델별 ROC 커브 비교
│   ├── 03_confusion_matrix_best.png  # 최종 모델 혼동 행렬
│   ├── 04_feature_importance.png     # SHAP 변수 중요도
│   └── EDA_*.png                     # EDA 시각화 차트
│
├── 01_Data_Preprocessing.ipynb       # 데이터 정제 · 결측치 · 파생변수
├── 02_Statistical_Analysis.ipynb     # Rao-Scott 검정 · 로지스틱 회귀
├── EDA.ipynb                         # 가중치 반영 탐색적 데이터 분석
│
├── train.py                          # ML 파이프라인 (학습·평가·임계값·저장)
├── app.py                            # Streamlit 웹 애플리케이션
├── requirements.txt
└── README.md
```

---

## 🚀 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 모델 학습 (선택)
python train.py

# 3. 웹 서비스 실행
streamlit run app.py
```

---

## 👥 팀 정보

| 항목 | 내용 |
|------|------|
| 프로젝트명 | 치아업(cheer up) |
| 과정 | AI 의료서비스 모델 개발 과정 5회차 |
| 기간 | 2026.03.03 ~ 2026.04.01 |


---

<div align="center">

데이터 출처: [질병관리청 청소년건강행태조사](https://www.kdca.go.kr) (제16차, 2020년)

</div>