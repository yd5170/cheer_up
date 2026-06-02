import pandas as pd
import numpy as np

# df = pd.read_sas('C:\dev\healthcare\semi_1\data\kyrbs2023.sas7bdat', format='sas7bdat')
df = pd.read_sas('./data/raw/kyrbs2020.sas7bdat', format='sas7bdat')

# 필요한 원시 변수만 먼저 추출
target_cols = [
    'SEX', 'GRADE', 'E_S_RCRD', 'E_SES', 
    'INT_SPWD_TM', 'INT_SPWK_TM',
    'INT_SP_OU_1', 'INT_SP_OU_2', 'INT_SP_OU_3', 'INT_SP_OU_4', 'INT_SP_OU_5',
    'INT_SP_OU_6', 'INT_SP_OU_7', 'INT_SP_OU_8', 'INT_SP_OU_9', 'INT_SP_OU_10',
    'M_GAD_1', 'M_GAD_2', 'M_GAD_3', 'M_GAD_4', 'M_GAD_5', 'M_GAD_6', 'M_GAD_7',
    'M_STR', 'M_SAD', 'M_SUI_CON',
    'O_SYMP1', 'O_SYMP2', 'O_SYMP3', 'O_SYMP4',
    'M_SLP_EN','W'
]
df = df[target_cols].copy()

# -------------------------------------------------------------------
# 파생 변수 생성 및 범주화 (순서 지정 적용)
# -------------------------------------------------------------------

# 1. gender (남자 -> 여자 순)
df['gender'] = pd.Categorical(
    df['SEX'].map({1: 'Male', 2: 'Female'}), 
    categories=['Male', 'Female'], 
    ordered=True
)

# 2. school (중학교 -> 고등학교 순)
df['school'] = df['GRADE'].apply(lambda x: 'Middle school' if x in [1, 2, 3] else ('High school' if x in [4, 5, 6] else np.nan))
df['school'] = pd.Categorical(
    df['school'], 
    categories=['Middle school', 'High school'], 
    ordered=True
)

# 3. Grade (학업성적: 상 -> 중 -> 하 순)
df['grade'] = pd.Categorical(
    df['E_S_RCRD'].map({1: 'High', 2: 'High', 3: 'Middle', 4: 'Low', 5: 'Low'}),
    categories=['High', 'Middle', 'Low'], 
    ordered=True
)

# 4. Income (경제수준: 상 -> 중 -> 하 순)
df['income'] = pd.Categorical(
    df['E_SES'].map({1: 'High', 2: 'High', 3: 'Middle', 4: 'Low', 5: 'Low'}),
    categories=['High', 'Middle', 'Low'], 
    ordered=True
)

# 5. 스마트폰 사용 시간 범주화
def categorize_time(minutes):
    if pd.isna(minutes): return np.nan
    hours = minutes / 60
    if hours <= 3: return '≤3'
    elif 3 < hours <= 5: return '3 ~ 5'
    elif 5 < hours <= 8: return '5 ~ 8'
    else: return '≥8'

# 시간도 작은 것에서 큰 순서대로 정렬되도록 지정
time_categories = ['≤3', '3 ~ 5', '5 ~ 8', '≥8']
df['smartphone_use_day'] = pd.Categorical(
    df['INT_SPWD_TM'].apply(categorize_time),
    categories=time_categories, ordered=True
)
df['smartphone_use_weekend'] = pd.Categorical(
    df['INT_SPWK_TM'].apply(categorize_time),
    categories=time_categories, ordered=True
)

# 6. SmartPhone Dependence (문항 합산 후 분류)
sp_cols = [f'INT_SP_OU_{i}' for i in range(1, 11)]
df['sp_score'] = df[sp_cols].sum(axis=1)
df['smartphone_dependence'] = pd.Categorical(
    df['sp_score'].apply(lambda x: 'No' if x < 23 else 'Risk'),
    categories=['No', 'Risk'], ordered=True
)

# 7. Anxiety (각 문항에서 1씩 뺀 후 합산)
gad_cols = [f'M_GAD_{i}' for i in range(1, 8)]
df['anxiety_score'] = df[gad_cols].apply(lambda x: x - 1).sum(axis=1)

def categorize_anxiety(score):
    if pd.isna(score): return np.nan
    if score <= 4: return 'No'
    elif score <= 9: return 'Mild'
    elif score <= 14: return 'Moderate'
    else: return 'Severe'

# 불안 증세도 약한 것에서 심한 순서대로 정렬
df['anxiety'] = pd.Categorical(
    df['anxiety_score'].apply(categorize_anxiety),
    categories=['No', 'Mild', 'Moderate', 'Severe'], ordered=True
)

# 8. stress (스트레스: 상 -> 중 -> 하 순)
df['stress'] = pd.Categorical(
    df['M_STR'].map({1: 'High', 2: 'High', 3: 'Middle', 4: 'Low', 5: 'Low'}), 
    categories=['High', 'Middle', 'Low'], ordered=True
)

# 9. despair & suicidal thoughts
df['despair'] = pd.Categorical(df['M_SAD'].map({1: 'No', 2: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['suicidal_thoughts'] = pd.Categorical(df['M_SUI_CON'].map({1: 'No', 2: 'Yes'}), categories=['No', 'Yes'], ordered=True)

# 10. 구강 건강 관련 변수
df['tooth_fracture'] = pd.Categorical(df['O_SYMP1'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['chewing_discomfort'] = pd.Categorical(df['O_SYMP2'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['tooth_pain'] = pd.Categorical(df['O_SYMP3'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['gingival_bleeding'] = pd.Categorical(df['O_SYMP4'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)

df['oral_health'] = df[['O_SYMP1', 'O_SYMP2', 'O_SYMP3', 'O_SYMP4']].sum(axis=1)


df['oral_health'] = pd.Categorical(
    df['oral_health'].map({0 : 'No', 1 : 'Yes', 2 : 'Yes', 3 : 'Yes', 4 : 'Yes'}),
    categories=['No', 'Yes'], 
    ordered=True
)

# 11. 수면 질
df['sleep_quality'] = pd.Categorical(
    df['M_SLP_EN'].map({
        1: 'No', 2: 'No',           # 충분 (문제 없음)
        3: 'Yes', 4: 'Yes', 5: 'Yes' # 보통 혹은 부족 (문제 있음)
    }),
    categories=['No', 'Yes'], ordered=True
)

# -------------------------------------------------------------------
# 최종 데이터셋 구성
# -------------------------------------------------------------------
final_cols = [
    'gender', 'school', 'grade', 'income', 
    'smartphone_use_day', 'smartphone_use_weekend', 
    'smartphone_dependence', 
    'anxiety', 'stress', 'despair', 'suicidal_thoughts',
    'oral_health',
    'sleep_quality',
    'W'
]
df_final = df[final_cols]

# 데이터셋 정보 확인
df_final.info()

# 결측치 확인
print(df_final.isnull().sum())

# 결측치 처리
df_clean = df_final.dropna().copy()

print(f"제거 전 데이터 수: {len(df_final)}")
print(f"제거 후 데이터 수: {len(df_clean)}") # 결측치 제거 후 50975값으로 선행 논문과 동일한 수치

# df_clean.to_csv('C:/dev/healthcare/semi_1/data/kyrbs2023_clean_v1.csv', index=False, encoding='utf-8-sig')
df_clean.to_csv('./data/processed/kyrbs2020_clean_v1.csv', index=False, encoding='utf-8-sig')

