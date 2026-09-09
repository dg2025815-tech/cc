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
    try:
        # 데이터 불러오기 (UTF-8 인코딩)
        df = pd.read_csv(DATA_URL, encoding="utf-8")
        
        # '날짜' 열을 datetime 형으로 변환 및 연도 추출 (변환 불가 항목은 NaT 처리)
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df["연도"] = df["날짜"].dt.year
        
        # '평균기온' 및 '연도' 결측치 제거
        df = df.dropna(subset=["평균기온", "연도"])
        df["연도"] = df["연도"].astype(int)
        
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
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df_clean = load_and_process_data()

if df_clean.empty:
    st.warning("데이터를 불러오지 못했습니다. 인터넷 연결 상태나 데이터 URL을 확인해 주세요.")
    st.stop()

# 2. 회귀 모델 및 100년당 기온 상승량 계산
# (1) 전체 기간
x_all = df_clean["연도"].values.astype(float)
y_all = df_clean["연평균기온"].values.astype(float)

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
rate_100y_all = float(slope_all * 100)  # 100년 동안의 기온 변화량 (°C)
corr_coef_all = float(np.corrcoef(x_all, y_all)[0, 1])

# (2) 최근 20년 (마지막 연도 기준 최근 20년)
max_year = int(x_all.max())
df_recent = df_clean[df_clean["연도"] >= (max_year - 19)].copy()
x_recent = df_recent["연도"].values.astype(float)
y_recent = df_recent["연평균기온"].values.astype(float)

slope_recent, intercept_recent = np.polyfit(x_recent, y_recent, 1)
rate_100y_recent = float(slope_recent * 100)  # 100년 환산 기온 변화량 (°C)

# 학습 데이터 기본 정보
num_years = int(len(df_clean))
start_year = int(x_all.min())
end_year = int(x_all.max())

# 3. 사이드바 / UI 컨트롤
st.sidebar.header("🎛️ 예측 연도 선택")
target_year = st.sidebar.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)

# 전체 기간 회귀식 기준 선택 연도 예상 기온
predicted_temp = float(slope_all * target_year + intercept_all)

# 4. 화면 구성: 100년당 온난화 속도 비교 (큰 지표)
st.subheader("🔥 서울 기온 상승 속도 비교 (100년당 상승량)")

col_rate1, col_rate2, col_rate3 = st.columns(3)
col_rate1.metric(
    label=f"전체 기간 ({start_year}~{end_year}년)",
    value=f"+{rate_100y_all:.2f} °C / 100년",
    help="전체 데이터를 기반으로 계산한 100년당 평균기온 상승량입니다."
)
col_rate2.metric(
    label=f"최근 20년 ({max_year-19}~{max_year}년)",
    value=f"+{rate_100y_recent:.2f} °C / 100년",
    delta=f"{rate_100y_recent - rate_100y_all:.2f} °C (전체 대비 가속)",
    help="최근 20년 데이터를 추세선화하여 100년 단위로 환산한 상승량입니다."
)
col_rate3.metric(
    label=f"선택 연도({target_year}년) 예상 기온",
    value=f"{predicted_temp:.2f} °C",
    help="전체 기간 회귀 모델 기준 예측값입니다."
)

st.markdown("---")

# 학습 데이터 및 상관계수 정보
col_info1, col_info2 = st.columns(2)
col_info1.info(
    f"📊 **학습 데이터 정보**: 총 **{num_years}개** 연도 데이터 사용 "
    f"(시작: **{start_year}년** / 끝: **{end_year}년**)"
)
col_info2.success(
    f"📈 **전체 기간 상관계수 (r)**: **{corr_coef_all:.4f}** (양의 상관관계)"
)

# 5. Plotly 그래프 작성 (두 회귀선 함께 표시)
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(go.Scatter(
    x=x_all,
    y=y_all,
    mode='markers',
    name='실제 연평균기온',
    marker=dict(color='deepskyblue', size=8)
))

# 전체 기간 회귀 직선 (1900~2100년)
x_line = np.linspace(1900, 2100, 201)
y_line_all = slope_all * x_line + intercept_all
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line_all,
    mode='lines',
    name=f'전체 기간 추세선 (+{rate_100y_all:.2f}°C/100년)',
    line=dict(color='firebrick', width=2.5, dash='dash')
))

# 최근 20년 회귀 직선 (1900~2100년)
y_line_recent = slope_recent * x_line + intercept_recent
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line_recent,
    mode='lines',
    name=f'최근 20년 추세선 (+{rate_100y_recent:.2f}°C/100년)',
    line=dict(color='orange', width=2, dash='dot')
))

# 선택한 연도 예측점 강조
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
    title="서울 연도별 평균기온 추이 및 추세선 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
