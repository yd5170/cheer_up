청소년 스마트폰 의존과 수면의 질에 따른 구강 건강 위험도 예측 

- 프로젝트 개요
본 프로젝트는 '제16차 청소년건강행태조사(KYRBS 2020)' 데이터를 활용하여, 청소년의 스마트폰 과의존과 수면의 질이 구강 건강에 미치는 영향을 분석하고 위험도를 예측하는 머신러닝 파이프라인 및 웹 서비스입니다. 

단순한 질병 예측을 넘어, 복합표본설계 가중치(W)를 전면 적용한 정교한 통계 검정을 수행하였으며, 분석 결과를 바탕으로 개인 맞춤형 행동 개선 가이드와 의료기관 연계 솔루션을 제공하는 '통합 헬스케어 플랫폼'을 제안합니다.

- 주요 기능 및 특징
1. 정교한 통계 및 탐색적 분석 (EDA)
   - 복합표본설계 가중치를 반영한 다중 로지스틱 회귀분석 및 Rao-Scott 카이제곱 검정 수행
   - 스마트폰 의존도와 수면의 질의 상호작용이 구강 건강에 미치는 영향 시각화
2. 머신러닝 예측 모델 최적화
   - 질병 예측(의료/헬스케어) 도메인 특성에 맞춰 '재현율(Recall ≈ 0.85)' 극대화를 목표로 모델 학습
   - Optuna를 활용한 하이퍼파라미터 튜닝 및 Youden's Index 기반 최적 임계값 도출 (XGBoost 최종 선정)
3. 사용자 친화적 웹 대시보드 (Streamlit)
   - S-Scale(스마트폰 과의존 척도) 자가진단 및 XGBoost 기반 구강 건강 위험도 실시간 예측
   - 진단 결과(4가지 그룹)에 따른 1:1 맞춤형 생활습관/구강 관리 가이드라인 제공
   - 주변 치과 검색 및 예약 연계 시스템 구현

디렉토리 구조
```text
Oral_Health_Prediction/
│
├── data/                       # 데이터 폴더
│   ├── raw/                    # 원본 데이터 (KYRBS 2020 sas7bdat)
│   ├── processed/              # 전처리 완료 데이터 (csv)
│   └── 논문, 원시자료지침서/     # 참고 문헌 및 지침서
│
├── models/                     # 학습된 머신러닝 모델 및 메타데이터
│   ├── xgboost_model.pkl       # 최종 선정된 XGBoost 모델
│   ├── scaler.pkl              # Weighted StandardScaler 
│   ├── model_meta.json         # 변수 매핑 및 Youden's Index 임계값 정보
│   └── best_params.json        # Optuna 튜닝 결과
│
├── plots/                      # 분석 및 평가 시각화 이미지
│   ├── 01_roc_curves.png       # 모델별 ROC 커브 비교
│   ├── 03_confusion_matrix_best.png 
│   ├── 04_feature_importance.png
│   └── EDA_...png              # 다양한 탐색적 데이터 분석 차트
│
├── 01_Data_Preprocessing.ipynb # 데이터 정제 및 결측치/파생변수 처리
├── 02_Statistical_Analysis.ipynb # 전통적 통계 분석 (위계적 로지스틱 회귀 등)
├── EDA.ipynb                   # 가중치 반영 탐색적 데이터 분석 (시각화)
│
├── train.py                    # ML 파이프라인 (학습, 평가, 임계값 조정, 저장)
├── app.py                     # Streamlit 웹 애플리케이션 프론트엔드
├── requirements.txt         # 파이썬 라이브러리 및 패키지 의존성 목록                     
└── README.md              # 프로젝트 설명서