import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 예측기")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    # 데이터 불러오기 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # '날짜' 열을 datetime 형으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 결측치 제거 (평균기온 기준)
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측 일수 및 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_stats[
        (yearly_stats["연도"] <= 2025) & (yearly_stats["관측일수"] >= 300)
    ].copy()
    
    return filtered_df

df_clean = load_and_process_data()

# 2. 선형 회귀 모델 계산 (Numpy polyfit)
x = df_clean["연도"].values
y = df_clean["연평균기온"].values

# 1차 회귀 직선 기울기(slope)와 절편(intercept)
slope, intercept = np.polyfit(x, y, 1)

# 상관계수 계산
corr_coef = np.corrcoef(x, y)[0, 1]

# 데이터 정보 추출
num_years = len(df_clean)
start_year = int(x.min())
end_year = int(x.max())

# 3. 사이드바 / UI 컨트롤
st.sidebar.header("🎛️ 예측 연도 선택")
target_year = st.sidebar.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)

# 선택한 연도의 예상 기온 계산
predicted_temp = slope * target_year + intercept

# 4. 상단 지표 출력
col1, col2, col3 = st.columns(3)
col1.metric("선택한 연도", f"{target_year}년")
col2.metric("예상 평균기온", f"{predicted_temp:.2f} °C")
col3.metric("상관계수 (r)", f"{corr_coef:.4f}")

st.markdown("---")

# 학습 데이터 정보 안내
st.info(
    f"📊 **학습 데이터 정보**: 총 **{num_years}개**의 연도 데이터를 사용하였습니다. "
    f"(시작 연도: **{start_year}년**, 끝 연도: **{end_year}년**)"
)

# 5. Plotly 그래프 그리기
fig = go.Figure()

# (1) 실제 연평균기온 산점도
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    name='실제 연평균기온',
    marker=dict(color='deepskyblue', size=8)
))

# (2) 회귀 직선 (1900년부터 2100년까지 확장하여 시각화)
x_line = np.linspace(1900, 2100, 201)
y_line = slope * x_line + intercept

fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode='lines',
    name='선형 회귀 직선',
    line=dict(color='firebrick', width=2, dash='dash')
))

# (3) 선택한 연도 예측점 강조
fig.add_trace(go.Scatter(
    x=[target_year],
    y=[predicted_temp],
    mode='markers+text',
    name=f'{target_year}년 예측점',
    text=[f"{predicted_temp:.2f}°C"],
    textposition="top center",
    marker=dict(color='gold', size=14, symbol='star')
))

# Layout 설정
fig.update_layout(
    title="서울 연도별 연평균기온 추이 및 추세선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
