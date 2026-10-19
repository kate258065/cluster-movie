import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 클러스터링)")
st.write(
    "영화 데이터를 바탕으로 비지도 학습(K-Means)을 활용하여 영화를 유형별로 묶어주는 앱입니다."
)


# 2. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 결측치 제거 및 필수 조건 처리 (첫 주 관객 > 0)
    required_cols = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi",
        "movieNm",
    ]
    df = df.dropna(subset=required_cols)
    df = df[df["first_week_audi"] > 0].copy()

    # 파생 변수 생성
    # 1) 스크린 수 상용로그 (log10(first_scrn + 1))
    df["log_first_scrn"] = np.log10(df["first_scrn"] + 1)
    # 2) 누적 관객 상용로그 (log10(total_audi + 1))
    df["log_total_audi"] = np.log10(df["total_audi"] + 1)
    # 3) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
    df["long_run_index"] = (df["total_audi"] / df["first_week_audi"]).clip(
        upper=20
    )

    return df


# 원래 데이터 로드 (결측치 처리 전 전체 개수 파악용)
raw_df = pd.read_csv(
    "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv",
    encoding="utf-8",
)
df = load_data()

st.write(
    f"**전체 영화 수:** {len(raw_df):,}편 | **분석 대상(조건에 맞는) 영화 수:** {len(df):,}편"
)

st.divider()

# 3. 군집화 속성 선택
feature_dict = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

selected_feature_names = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_dict[name] for name in selected_feature_names]

# 4. K-Means 클러스터링 수행
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# 난수 고정 (random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)

# 누적 관객(total_audi) 평균을 기준으로 묶음 라벨 재정렬 (큰 순서대로 ㉮, ㉯, ㉰)
df_temp = df.copy()
df_temp["raw_cluster"] = raw_clusters
cluster_audi_mean = (
    df_temp.groupby("raw_cluster")["total_audi"].mean().sort_values(ascending=False)
)

label_mapping = {}
labels = ["㉮", "㉯", "㉰"]
for idx, original_cluster_id in enumerate(cluster_audi_mean.index):
    label_mapping[original_cluster_id] = labels[idx]

df["cluster"] = [label_mapping[c] for c in raw_clusters]

# 5. 시각화 (2차원 및 3차원 산점도)
st.subheader("📊 클러스터 시각화")

col1, col2 = st.columns(2)

# --- 2차원 산점도 ---
st.markdown("### 2차원 산점도")
c1, c2 = st.columns(2)
with c1:
    x_axis_2d = st.selectbox(
        "2D 가로축(X):",
        options=selected_feature_names,
        index=0,
        key="2d_x",
    )
with c2:
    y_axis_2d = st.selectbox(
        "2D 세로축(Y):",
        options=selected_feature_names,
        index=min(1, len(selected_feature_names) - 1),
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_axis_2d],
    y=feature_dict[y_axis_2d],
    color="cluster",
    hover_name="movieNm",
    labels={
        feature_dict[x_axis_2d]: x_axis_2d,
        feature_dict[y_axis_2d]: y_axis_2d,
        "cluster": "영화 묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    title=f"2D 산점도 ({x_axis_2d} vs {y_axis_2d})",
)
st.plotly_chart(fig_2d, use_container_width=True)

# --- 3차원 산점도 ---
st.markdown("### 3차원 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 그리려면 선택된 속성이 3개 이상이어야 합니다.")
else:
    c3, c4, c5 = st.columns(3)
    with c3:
        x_axis_3d = st.selectbox(
            "3D X축:",
            options=selected_feature_names,
            index=0,
            key="3d_x",
        )
    with c4:
        y_axis_3d = st.selectbox(
            "3D Y축:",
            options=selected_feature_names,
            index=1,
            key="3d_y",
        )
    with c5:
        z_axis_3d = st.selectbox(
            "3D Z축:",
            options=selected_feature_names,
            index=2,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_axis_3d],
        y=feature_dict[y_axis_3d],
        z=feature_dict[z_axis_3d],
        color="cluster",
        hover_name="movieNm",
        labels={
            feature_dict[x_axis_3d]: x_axis_3d,
            feature_dict[y_axis_3d]: y_axis_3d,
            feature_dict[z_axis_3d]: z_axis_3d,
            "cluster": "영화 묶음",
        },
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        title=f"3D 산점도 ({x_axis_3d} vs {y_axis_3d} vs {z_axis_3d})",
    )
    # 점 크기 조절 (작게 설정)
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 6. 묶음별 통계 및 대표 영화 5편 출력
st.subheader("📋 묶음별 통계 및 대표 영화")

# 묶음별 편수 및 original unit 속성 평균 계산
stats_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieCd", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        TOP10_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("long_run_index", "mean"),
    )
    .reindex(labels)
)

# 숫자를 보기 쉽게 포맷팅
stats_df_display = stats_df.copy()
stats_df_display["편수"] = stats_df_display["편수"].map("{:,}편".format)
stats_df_display["스크린_수_평균"] = stats_df_display["스크린_수_평균"].map(
    "{:,.1f}개".format
)
stats_df_display["누적_관객_평균"] = stats_df_display["누적_관객_평균"].map(
    "{:,.0f}명".format
)
stats_df_display["TOP10_일수_평균"] = stats_df_display[
    "TOP10_일수_평균"
].map("{:.1f}일".format)
stats_df_display["롱런_지수_평균"] = stats_df_display["롱런_지수_평균"].map(
    "{:.2f}".format
)

st.markdown("#### 1. 묶음별 속성 평균 (원래 단위)")
st.dataframe(stats_df_display, use_container_width=True)

st.markdown("#### 2. 묶음별 누적 관객수 TOP 5 영화")
top_cols = st.columns(3)

for idx, label in enumerate(labels):
    with top_cols[idx]:
        st.markdown(f"**영화 묶음 {label}**")
        top_movies = (
            df[df["cluster"] == label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)[["movieNm", "total_audi"]]
        )

        for i, (_, row) in enumerate(top_movies.iterrows(), 1):
            st.write(f"{i}. **{row['movieNm']}** ({row['total_audi']:,}명)")
