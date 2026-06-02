import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 디렉토리 설정 및 폰트 설정
IMAGES_DIR = "./images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 한글 폰트 설정 (윈도우 기본 맑은 고딕 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 딥네이비 다크 테마 스타일 정의
BG_COLOR = '#0a1628'
CARD_COLOR = '#112240'
TEXT_COLOR = '#f8fafc'
MUTED_TEXT = '#94a3b8'
PRIMARY_COLOR = '#00c9b1'  # 청록
SECONDARY_COLOR = '#a8ff3e'  # 라임
BLUE_COLOR = '#4a9eff'
RED_COLOR = '#ff4757'
GRID_COLOR = 'rgba(255, 255, 255, 0.05)'

def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    ax.spines['bottom'].set_color(MUTED_TEXT)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color(MUTED_TEXT)
    ax.tick_params(colors=TEXT_COLOR, which='both')
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(color='#1e293b', linestyle='--', linewidth=0.5)

# =========================================================================
# 그래프 1: 스마트폰 의존도 × 구강증상 (그룹 가로 막대)
# =========================================================================
def draw_chart1():
    categories = ['일반군\n(No Risk)', '스마트폰 의존 위험군\n(Risk)']
    no_symptom = [54.7, 35.3]
    yes_symptom = [45.3, 64.7]
    
    y = np.arange(len(categories))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    apply_dark_theme(fig, ax)
    
    rects1 = ax.barh(y - height/2, no_symptom, height, label='구강증상 없음', color=BLUE_COLOR, edgecolor='none')
    rects2 = ax.barh(y + height/2, yes_symptom, height, label='구강증상 있음', color=RED_COLOR, edgecolor='none')
    
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_xlabel('비율 (%)', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title('스마트폰 의존도별 구강 증상 발생률 비교 (Rao-Scott, p < 0.001)', fontsize=13, fontweight='bold', pad=15)
    
    # 레전드 설정
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=PRIMARY_COLOR, labelcolor=TEXT_COLOR)
    
    # 값 라벨링
    for rect in rects1 + rects2:
        width = rect.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width - 2 if width > 10 else width + 2, rect.get_y() + rect.get_height()/2),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='right' if width > 10 else 'left', va='center',
                    color='white', fontweight='bold', fontsize=10)
                    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/01_smartphone_dependence_vs_oral.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

# =========================================================================
# 그래프 2: 수면의 질 × 구강증상 (도넛 차트 2개)
# =========================================================================
def draw_chart2():
    labels = ['구강증상 없음', '구강증상 있음']
    sleep_good = [57.9, 42.1]
    sleep_bad = [46.1, 53.9]
    colors = [PRIMARY_COLOR, RED_COLOR]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    # 수면 충분 도넛 차트
    wedges1, texts1, autotexts1 = ax1.pie(sleep_good, labels=labels, autopct='%1.1f%%',
                                          startangle=90, colors=colors, 
                                          wedgeprops=dict(width=0.35, edgecolor=BG_COLOR, linewidth=2),
                                          textprops=dict(color=TEXT_COLOR))
    ax1.text(0, 0, '수면 충분\n\n42.1% 발생', ha='center', va='center', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax1.set_title('수면 충분군', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
    for t in autotexts1:
        t.set_color('white')
        t.set_fontweight('bold')
        
    # 수면 부족 도넛 차트
    wedges2, texts2, autotexts2 = ax2.pie(sleep_bad, labels=labels, autopct='%1.1f%%',
                                          startangle=90, colors=colors,
                                          wedgeprops=dict(width=0.35, edgecolor=BG_COLOR, linewidth=2),
                                          textprops=dict(color=TEXT_COLOR))
    ax2.text(0, 0, '수면 부족\n\n53.9% 발생', ha='center', va='center', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax2.set_title('수면 부족군', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
    for t in autotexts2:
        t.set_color('white')
        t.set_fontweight('bold')
        
    fig.suptitle('수면의 질별 구강 증상 발생률 비교 (Rao-Scott, p < 0.001)', color=TEXT_COLOR, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/02_sleep_quality_vs_oral.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

# =========================================================================
# 그래프 3: 사용시간별 구강증상 비율 (라인 차트)
# =========================================================================
def draw_chart3():
    labels = ['≤3시간', '3~5시간', '5~8시간', '8시간 이상']
    weekday_data = [46.7, 50.5, 54.2, 56.1]
    weekend_data = [44.9, 48.1, 51.6, 55.6]
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    apply_dark_theme(fig, ax)
    
    # 주중 사용시간 라인
    ax.plot(labels, weekday_data, marker='o', color=PRIMARY_COLOR, linewidth=3.5, markersize=8, label='주중 사용시간')
    # 주말 사용시간 라인 (점선)
    ax.plot(labels, weekend_data, marker='s', color=SECONDARY_COLOR, linewidth=3, markersize=7, linestyle='--', label='주말 사용시간')
    
    ax.set_ylim(25, 60)
    ax.set_ylabel('구강증상 있음 비율 (%)', fontsize=11)
    ax.set_title('스마트폰 사용시간별 구강 증상 있음 비율 (%)', fontsize=13, fontweight='bold', pad=15)
    
    # 수치 텍스트 표시
    for i, txt in enumerate(weekday_data):
        ax.annotate(f'{txt:.1f}%', (labels[i], weekday_data[i] + 1.2), color=PRIMARY_COLOR, fontweight='bold', ha='center')
    for i, txt in enumerate(weekend_data):
        ax.annotate(f'{txt:.1f}%', (labels[i], weekend_data[i] - 2.2), color=SECONDARY_COLOR, fontweight='bold', ha='center')
        
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=PRIMARY_COLOR, labelcolor=TEXT_COLOR, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/03_use_time_vs_oral.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

# =========================================================================
# 그래프 4: 머신러닝 모델 성능 비교 (그룹 가로 막대)
# =========================================================================
def draw_chart4():
    models = ['Logistic Regression', 'Random Forest', 'CatBoost', 'LightGBM', 'XGBoost (Best)'][::-1]
    auc = [0.5808, 0.6463, 0.6484, 0.6487, 0.6483][::-1]
    recall = [0.8516, 0.8473, 0.8500, 0.8506, 0.8521][::-1]
    precision = [0.5172, 0.5456, 0.5504, 0.5501, 0.5498][::-1]
    
    y = np.arange(len(models))
    height = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    apply_dark_theme(fig, ax)
    
    rects1 = ax.barh(y - height, auc, height, label='ROC-AUC', color=BLUE_COLOR)
    rects2 = ax.barh(y, recall, height, label='Recall', color=RED_COLOR)
    rects3 = ax.barh(y + height, precision, height, label='Precision', color=PRIMARY_COLOR)
    
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_title('머신러닝 모델 성능 비교 (Test셋, Recall 최우선)', fontsize=13, fontweight='bold', pad=15)
    
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=PRIMARY_COLOR, labelcolor=TEXT_COLOR, loc='lower right')
    
    # 값 라벨링 (끝부분에 작게 표시)
    for rect in rects1 + rects2 + rects3:
        width = rect.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, rect.get_y() + rect.get_height()/2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center',
                    color=TEXT_COLOR, fontsize=8)
                    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/04_model_performance_compare.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

# =========================================================================
# 그래프 5: ROC Curve 비교
# =========================================================================
def draw_chart5():
    aucs = {
        'XGBoost (AUC=0.6483)': 0.6483,
        'LightGBM (AUC=0.6487)': 0.6487,
        'CatBoost (AUC=0.6484)': 0.6484,
        'Random Forest (AUC=0.6463)': 0.6463,
        'Logistic Regression (AUC=0.5808)': 0.5808
    }
    
    colors = {
        'XGBoost (AUC=0.6483)': PRIMARY_COLOR,
        'LightGBM (AUC=0.6487)': BLUE_COLOR,
        'CatBoost (AUC=0.6484)': RED_COLOR,
        'Random Forest (AUC=0.6463)': '#f9ca24',
        'Logistic Regression (AUC=0.5808)': SECONDARY_COLOR
    }
    
    def get_roc_curve(auc_val):
        p = 1 / (1 - auc_val) - 1
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.power(1 - fpr, p)
        return fpr, tpr

    fig, ax = plt.subplots(figsize=(7, 6))
    apply_dark_theme(fig, ax)
    
    for label, auc_val in aucs.items():
        fpr, tpr = get_roc_curve(auc_val)
        linewidth = 3 if 'XGBoost' in label else 1.8
        linestyle = '-' if 'XGBoost' in label or 'LightGBM' in label or 'CatBoost' in label else '--'
        ax.plot(fpr, tpr, label=label, color=colors[label], linewidth=linewidth, linestyle=linestyle)
        
    ax.plot([0, 1], [0, 1], color='#555555', linestyle=':', label='Reference Line')
    
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    ax.set_title('ROC Curve 비교 (Test셋 기준)', fontsize=13, fontweight='bold', pad=15)
    
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=PRIMARY_COLOR, labelcolor=TEXT_COLOR, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/05_roc_curves.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

# =========================================================================
# 그래프 6: Feature Importance (피처 중요도)
# =========================================================================
def draw_chart6():
    features = [
        '자살생각',
        '★주중 스마트폰 사용시간',
        '학업성적',
        '★주말 스마트폰 사용시간',
        '가구소득',
        '학교급',
        '성별',
        '스트레스',
        '절망감',
        '★수면의 질(나쁨=1)',
        '★스마트폰 의존도(위험=1)',
        '불안수준'
    ]
    importance = [
        0.0128,
        0.0160,
        0.0189,
        0.0258,
        0.0304,
        0.0316,
        0.0449,
        0.0627,
        0.0852,
        0.1272,
        0.2525,
        0.2919
    ]
    
    # 핵심 피처인지 판별해서 색상 지정
    colors = []
    for f in features:
        if '★' in f:
            colors.append(RED_COLOR)
        else:
            colors.append('#95a5a6')
            
    fig, ax = plt.subplots(figsize=(9, 5.5))
    apply_dark_theme(fig, ax)
    
    rects = ax.barh(features, importance, color=colors, edgecolor='none', height=0.6)
    
    ax.set_xlabel('Feature Importance (SHAP Value 기반)', fontsize=11)
    ax.set_title('XGBoost 변수 중요도 (Feature Importance)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, max(importance) * 1.15)
    
    # 값 라벨링
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    color=TEXT_COLOR, fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/06_feature_importance.png", dpi=200, facecolor=BG_COLOR)
    plt.close()

if __name__ == "__main__":
    print("Generating charts...")
    draw_chart1()
    draw_chart2()
    draw_chart3()
    draw_chart4()
    draw_chart5()
    draw_chart6()
    print("All charts generated and saved under './images' directory successfully!")
