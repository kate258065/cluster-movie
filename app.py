
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 성적과 상영 특성을 바탕으로 세 가지 유형을 찾아봅니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher/"
    "data-share-/main/kobis_movies.csv"
)

# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {missing_columns}"
        )

    numeric_columns = [
        "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 로그와 나눗셈을 적용할 수 있는 값으로 변환
    df["스크린 수"] = df["first_scrn"].where(df["first_scrn"] > 0)
    df["누적 관객"] = df["total_audi"].where(df["total_audi"] > 0)

    # 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
    df["롱런 지수"] = (
        df["total_audi"] / df["first_week_audi"].where(
            df["first_week_audi"] > 0
        )
    ).clip(upper=20)

    # 군집화에 사용할 네 가지 속성
    df["로그 스크린 수"] = df["스크린 수"].map(
        lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
    )
    df["로그 누적 관객"] = df["누적 관객"].map(
        lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
    )

    feature_columns = [
        "로그 스크린 수",
        "로그 누적 관객",
        "days_in_top10",
        "롱런 지수"
    ]

    # 네 속성 중 결측값이 있거나 첫 주 관객이 0 이하인 영화 제외
    valid = (
        df[feature_columns].notna().all(axis=1)
        & df["first_week_audi"].gt(0)
        & df["first_scrn"].gt(0)
        & df["total_audi"].gt(0)
        & df["days_in_top10"].ge(0)
    )

    df = df.loc[valid].copy()
    df.rename(columns={"days_in_top10": "10위권 일수"}, inplace=True)

    # 군집별 평균을 원래 단위로 표시할 때 사용할 속성
    original_features = [
        "first_scrn", "total_audi", "10위권 일수", "롱런 지수"
    ]

    return df, feature_columns, original_features


try:
    df, feature_columns, original_features = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.stop()

# --------------------------------------------------
# 속성 선택
# --------------------------------------------------
st.subheader("1. 군집화에 사용할 속성")

feature_labels = {
    "로그 스크린 수": "스크린 수 (상용로그)",
    "로그 누적 관객": "누적 관객 (상용로그)",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

selected_features = st.multiselect(
    "군집화 속성을 두 개 이상 선택하세요.",
    options=feature_columns,
    default=feature_columns,
    format_func=lambda x: feature_labels[x]
)

# 전체 데이터 편수와 전처리 후 군집화 가능한 편수를 한 줄로 표시
st.markdown(
    f"**전체 영화:** {len(pd.read_csv(DATA_URL, encoding='utf-8')):,}편"
    f"　|　**군집화 대상:** {len(df):,}편"
)

if len(selected_features) < 2:
    st.warning("K-평균 군집화를 위해 속성을 두 개 이상 선택하세요.")
    st.stop()

# --------------------------------------------------
# 표준화 및 K-평균 군집화
# --------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집 번호"] = kmeans.fit_predict(scaled_data)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: f"㉮"[0] if rank == 0 else
                 f"㉯"[0] if rank == 1 else f"㉰"[0]
    for rank, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집 번호"].map(cluster_labels)

cluster_colors = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B"
}

# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: feature_labels[x],
        key="scatter_x"
    )

with axis_col2:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=min(1, len(selected_features) - 1),
        format_func=lambda x: feature_labels[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    color_discrete_map=cluster_colors,
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        x_feature: ":.3f",
        y_feature: ":.3f"
    },
    labels={
        x_feature: feature_labels[x_feature],
        y_feature: feature_labels[y_feature],
        "영화 유형": "영화 유형"
    },
    opacity=0.75
)

fig2.update_traces(marker={"size": 7})
fig2.update_layout(
    legend_title_text="영화 유형",
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택하세요.")
else:
    axis3_col1, axis3_col2, axis3_col3 = st.columns(3)

    with axis3_col1:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            format_func=lambda x: feature_labels[x],
            key="scatter3_x"
        )

    with axis3_col2:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            format_func=lambda x: feature_labels[x],
            key="scatter3_y"
        )

    with axis3_col3:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            format_func=lambda x: feature_labels[x],
            key="scatter3_z"
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        color_discrete_map=cluster_colors,
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f"
        },
        labels={
            x3: feature_labels[x3],
            y3: feature_labels[y3],
            z3: feature_labels[z3],
            "영화 유형": "영화 유형"
        },
        opacity=0.8
    )

    fig3.update_traces(marker={"size": 3})
    fig3.update_layout(
        scene=dict(
            xaxis_title=feature_labels[x3],
            yaxis_title=feature_labels[y3],
            zaxis_title=feature_labels[z3]
        ),
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# 군집별 통계
# --------------------------------------------------
st.subheader("4. 군집별 통계")

stats_rows = []

for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label]

    stats_rows.append({
        "영화 유형": label,
        "영화 편수": len(group),
        "평균 스크린 수": group["first_scrn"].mean(),
        "평균 누적 관객": group["total_audi"].mean(),
        "평균 10위권 일수": group["10위권 일수"].mean(),
        "평균 롱런 지수": group["롱런 지수"].mean()
    })

stats_df = pd.DataFrame(stats_rows)
stats_df = stats_df[stats_df["영화 편수"] > 0]

display_stats = stats_df.copy()
for col in [
    "평균 스크린 수",
    "평균 누적 관객",
    "평균 10위권 일수",
    "평균 롱런 지수"
]:
    display_stats[col] = display_stats[col].round(2)

st.dataframe(
    display_stats,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "평균은 원래 단위로 계산했습니다. "
    "스크린 수는 개, 관객 수는 명, 10위권 일수는 일 단위입니다."
)

# --------------------------------------------------
# 군집별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("5. 군집별 누적 관객 상위 5편")

for label in ["㉮", "㉯", "㉰"]:
    group = (
        df[df["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(f"{label} 유형 — 누적 관객 상위 영화", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top5 = group[["movieNm", "total_audi"]].copy()
            top5.columns = ["영화 제목", "누적 관객 수"]
            top5["누적 관객 수"] = top5["누적 관객 수"].map(
                lambda x: f"{x:,.0f}명"
            )
            top5.insert(0, "순위", range(1, len(top5) + 1))

            st.dataframe(
                top5,
                use_container_width=True,
                hide_index=True
            )
