import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정 (앱 제목, 아이콘, 브라우저 탭 제목)
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df


df_raw = load_data()

# 3. 데이터 전처리 및 피처 생성
# - 롱런 지수 = 누적 관객(total_audi) / 첫 주 관객(first_week_audi)
# - 네 속성 중 값이 없거나 첫 주 관객(first_week_audi)이 0인 영화 제외
df = df_raw.copy()

# 필요 컬럼 결측치 제거 및 첫 주 관객 > 0 조건 적용
df = df.dropna(
    subset=["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
)
df = df[df["first_week_audi"] > 0].copy()

# 피처 생성
df["log_first_scrn"] = np.log1p(df["first_scrn"])
df["log_total_audi"] = np.log1p(df["total_audi"])
df["long_run_index"] = df["total_audi"] / df["first_week_audi"]

# 전처리 후 전체 데이터 수
total_count = len(df)

# 속성 이름 매핑 (화면 표시용 <-> 내부 컬럼명)
feature_map = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}
inv_feature_map = {v: k for k, v in feature_map.items()}

# 4. 사용자 속성 선택 UI (기본 4개 모두 선택, 최소 2개 이상 선택)
st.subheader("⚙️ 클러스터링 설정")
selected_feature_names = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (2개 이상 선택 필수):",
    options=list(feature_map.keys()),
    default=list(feature_map.keys()),
)

if len(selected_feature_names) < 2:
    st.warning("속성을 2개 이상 선택해주세요.")
    st.stop()

selected_features = [feature_map[name] for name in selected_feature_names]

# 5. 데이터 표준화 및 K-평균 군집화 (k=3, random_state=42)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 6. 군집 번호 재할당 (누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰)
cluster_audi_mean = (
    df.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False)
)
rank_to_symbol = {0: "㉮", 1: "㉯", 2: "㉰"}

mapping = {}
for rank, (raw_cluster, _) in enumerate(cluster_audi_mean.items()):
    mapping[raw_cluster] = rank_to_symbol[rank]

df["cluster"] = df["cluster_raw"].map(mapping)

# 전체 편수와 묶은 편수를 한 줄로 작성
st.write(f"**전체 영화 편수:** {total_count}편 | **묶은 영화 편수:** {len(df)}편")

st.markdown("---")

# 7. 2차원 산점도
st.subheader("📊 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "2차원 X축 속성 선택:",
        options=selected_feature_names,
        index=0,
        key="2d_x",
    )
with col2:
    y_axis_2d = st.selectbox(
        "2차원 Y축 속성 선택:",
        options=selected_feature_names,
        index=min(1, len(selected_feature_names) - 1),
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_map[x_axis_2d],
    y=feature_map[y_axis_2d],
    color="cluster",
    hover_name="movieNm",
    labels={
        feature_map[x_axis_2d]: x_axis_2d,
        feature_map[y_axis_2d]: y_axis_2d,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    title=f"{x_axis_2d} vs {y_axis_2d} 2차원 산점도",
)
st.plotly_chart(fig_2d, use_container_width=True)

# 8. 3차원 산점도
st.subheader("🌐 3차원 산점도")
col3, col4, col5 = st.columns(3)
with col3:
    x_axis_3d = st.selectbox(
        "3차원 X축 속성 선택:",
        options=selected_feature_names,
        index=0,
        key="3d_x",
    )
with col4:
    y_axis_3d = st.selectbox(
        "3차원 Y축 속성 선택:",
        options=selected_feature_names,
        index=min(1, len(selected_feature_names) - 1),
        key="3d_y",
    )
with col5:
    z_axis_3d = st.selectbox(
        "3차원 Z축 속성 선택:",
        options=selected_feature_names,
        index=min(2, len(selected_feature_names) - 1),
        key="3d_z",
    )

fig_3d = px.scatter_3d(
    df,
    x=feature_map[x_axis_3d],
    y=feature_map[y_axis_3d],
    z=feature_map[z_axis_3d],
    color="cluster",
    hover_name="movieNm",
    labels={
        feature_map[x_axis_3d]: x_axis_3d,
        feature_map[y_axis_3d]: y_axis_3d,
        feature_map[z_axis_3d]: z_axis_3d,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    title="3차원 산점도 (마우스로 회전 가능)",
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 9. 묶음별 편수 및 네 속성의 평균(원래 단위) 표
st.subheader("📋 묶음별 통계 요약 (원래 단위)")

summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        일주일내_top10일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

st.dataframe(summary_df.round(2), use_container_width=True)

# 10. 묶음마다 누적 관객이 많은 순서로 영화 5편의 제목 표시
st.subheader("🏆 묶음별 누적 관객 TOP 5 영화")

top5_cols = st.columns(3)
clusters = ["㉮", "㉯", "㉰"]

for idx, cluster_symbol in enumerate(clusters):
    with top5_cols[idx]:
        st.markdown(f"### 묶음 {cluster_symbol}")
        cluster_movies = df[df["cluster"] == cluster_symbol]
        top5 = cluster_movies.sort_values(
            by="total_audi", ascending=False
        ).head(5)

        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            st.write(
                f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)"
            )
