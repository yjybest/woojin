import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── 데이터 로딩 ─────────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순으로 시도하고 바이트를 직접 읽어 빈 파일 오류 방지"""
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df
        except Exception:
            continue
    return pd.DataFrame()

def std_cols(df):
    """열 이름을 표준 이름으로 바꾸고 숫자 변환·생중량 자동 계산·시간 파싱 수행"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (pd.to_datetime(parts[0], format="%Y%m%d") +
                           pd.to_timedelta(parts[1].astype(int), unit="h"))
        df["time"] = parsed
    return df

def school_name_from_filename(fname):
    """파일명 앞부분에서 학교명 추출"""
    if not fname:
        return "송도고"
    base = os.path.splitext(fname)[0]
    m = re.match(r"^(.+?)(?:_|\d+[_\s].*)?$", base)
    return m.group(1) if m else base

def cols_in(df, names):
    return [c for c in names if c in df.columns]

def safe_mean(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.mean() if len(v) >= 1 else None

def safe_min(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.min() if len(v) >= 1 else None

def safe_max(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.max() if len(v) >= 1 else None

def safe_corr(x, y):
    df2 = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                        "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df2) < 2:
        return None
    return df2["x"].corr(df2["y"])

def trendline_trace(x, y, color="red"):
    """np.polyfit으로 추세선 go.Scatter 생성 (statsmodels 불필요)"""
    df2 = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                        "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df2) < 2:
        return None
    slope, intercept = np.polyfit(df2["x"], df2["y"], 1)
    xs = np.array([df2["x"].min(), df2["x"].max()])
    ys = slope * xs + intercept
    return go.Scatter(x=xs, y=ys, mode="lines",
                     line=dict(color=color, dash="dash", width=2),
                     name=f"추세선 (r={safe_corr(x,y):.3f})")

def download_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def base_layout(fig, title=None):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=title,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ── 사이드바 ────────────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")
env_file    = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

growth_items = ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]
sel_items    = st.sidebar.multiselect("생육 측정 항목 선택 (그래프에 표시)", growth_items,
                                      default=["잎 수", "지상부 길이", "생중량"])

# ── 데이터 로드 (한 번만) ────────────────────────────────────────
with st.spinner("데이터를 읽고 정리 중이에요..."):
    env_raw    = read_csv(env_file)
    growth_raw = read_csv(growth_file)
    env_df    = std_cols(env_raw)    if not env_raw.empty    else pd.DataFrame()
    growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

school_name = school_name_from_filename(
    env_file.name if env_file else (growth_file.name if growth_file else "")
)
st.sidebar.caption(f"📌 감지된 학교: **{school_name}**")
st.sidebar.divider()
st.sidebar.info("💡 CSV를 아직 업로드하지 않았다면, 파일 선택 후 탭 내용이 자동으로 표시됩니다.")

# ── 탭 ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ===================================================================
#  TAB 1 — 실험 개요
# ===================================================================
with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown("""
**극지식물**은 낮은 온도와 건조한 환경에서도 살아남는 강한 식물이에요.  
이 대시보드에서는 **우리 학교**에서 키운 극지식물 개체 하나하나가  
얼마나 다르게 자랐는지 비교하고, **온도·습도 같은 환경 조건**과의 관계도 살펴봅니다.

- **평균·표준편차·변동계수(CV):** 평균(산술평균)은 모든 값을 더해 개수로 나눈 값,  
  표준편차(SD)는 값이 평균 주변에 얼마나 모여 있는지를 나타내는 숫자예요.  
  CV(변동계수) = 표준편차 ÷ 평균 × 100(%)으로, 평균이 달라도 퍼짐 정도를 비교할 때 써요.
- **상관계수(r):** -1에서 +1 사이의 값으로, 두 값이 함께 증가하거나 함께 감소하는 경향을 나타내요.  
  0에 가까우면 뚜렷한 관계가 없다는 뜻이에요.
        """.strip())

        st.header("데이터 요약")
        env_cols_avail    = cols_in(env_df,    ["temperature", "humidity", "ph", "ec"])
        growth_cols_avail = cols_in(growth_df, ["잎 수", "지상부 길이", "지하부 길이",
                                                 "지상부 질량", "지하부 질량", "생중량"])
        n_growth = len(growth_df)

        period_text = "데이터 없음"
        if "time" in env_df.columns and env_df["time"].notna().any():
            tmin = env_df["time"].min()
            tmax = env_df["time"].max()
            period_text = f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"

        rows = []
        rows.append(["항목", "내용"])
        rows.append(["학교명", school_name])
        rows.append(["환경 데이터 측정 횟수", f"{len(env_df)} 행" if len(env_df) else "업로드 안 됨"])
        rows.append(["생육 데이터 개체 수", f"{n_growth} 개체" if n_growth else "업로드 안 됨"])
        rows.append(["측정 기간", period_text])
        env_desc = ", ".join(env_cols_avail) if env_cols_avail else "없음"
        rows.append(["환경 측정 항목", env_desc])
        gr_desc  = ", ".join(growth_cols_avail) if growth_cols_avail else "없음"
        rows.append(["생육 측정 항목", gr_desc])

        summary_df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(summary_df, width="stretch", hide_index=True)

        st.header("주요 지표 카드")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("총 개체 수",
                  f"{n_growth} 개체" if n_growth else "—")

        t_mean = safe_mean(env_df["temperature"]) if "temperature" in env_df.columns else None
        c2.metric("평균 온도 (°C)",
                  f"{t_mean:.1f}" if t_mean is not None else "—")

        h_mean = safe_mean(env_df["humidity"]) if "humidity" in env_df.columns else None
        c3.metric("평균 습도 (%)",
                  f"{h_mean:.1f}" if h_mean is not None else "—")

        bw_mean = safe_mean(growth_df["생중량"]) if "생중량" in growth_df.columns else None
        c4.metric("평균 생중량 (g)",
                  f"{bw_mean:.2f}" if bw_mean is not None else "—")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===================================================================
#  TAB 2 — 환경 데이터
# ===================================================================
with tab2:
    try:
        if env_df.empty:
            st.info("환경 CSV를 업로드하면 이 탭의 내용이 여기에 표시됩니다.")
        else:
            st.header("🌡️ 환경 데이터 요약")
            env_spec_cols = cols_in(env_df, ["temperature", "humidity", "ph", "ec"])
            if env_spec_cols:
                summ_rows = [["측정 항목", "평균", "최소", "최대"]]
                label_map = {"temperature": "온도 (°C)", "humidity": "습도 (%)",
                             "ph": "pH", "ec": "EC (mS/cm)"}
                for col in env_spec_cols:
                    summ_rows.append([
                        label_map.get(col, col),
                        f"{safe_mean(env_df[col]):.2f}" if safe_mean(env_df[col]) is not None else "—",
                        f"{safe_min(env_df[col]):.2f}" if safe_min(env_df[col]) is not None else "—",
                        f"{safe_max(env_df[col]):.2f}" if safe_max(env_df[col]) is not None else "—",
                    ])
                st.dataframe(pd.DataFrame(summ_rows[1:], columns=summ_rows[0]),
                             width="stretch", hide_index=True)

            st.header("날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns:
                env_df_day = env_df.copy()
                env_df_day["day"] = env_df_day["time"].dt.floor("D")

                def daily_line(col, ylabel, color):
                    if col not in env_df.columns or env_df[col].dropna().empty:
                        return None
                    grp = env_df_day.groupby("day")[col].mean().dropna()
                    if len(grp) < 2:
                        return None
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=grp.index, y=grp.values,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        name=ylabel,
                        marker=dict(size=5)
                    ))
                    base_layout(fig, title=f"{ylabel} 일별 평균")
                    fig.update_xaxes(title="날짜", tickformat="%m-%d")
                    fig.update_yaxes(title=ylabel)
                    return fig

                t_fig = daily_line("temperature", "온도 (°C)", "#ff6b6b")
                h_fig = daily_line("humidity", "습도 (%)", "#4ecdc4")

                has_ph = "ph" in env_df.columns
                has_ec = "ec" in env_df.columns

                if has_ph or has_ec:
                    fig_mix = make_subplots(rows=1, cols=2,
                                            subplot_titles=("pH 변화", "EC 변화"))
                    if has_ph:
                        g = env_df_day.groupby("day")["ph"].mean().dropna()
                        if len(g) >= 2:
                            fig_mix.add_trace(
                                go.Scatter(x=g.index, y=g.values,
                                           mode="lines+markers",
                                           line=dict(color="#9b59b6", width=2),
                                           name="pH"), row=1, col=1)
                    if has_ec:
                        g = env_df_day.groupby("day")["ec"].mean().dropna()
                        if len(g) >= 2:
                            fig_mix.add_trace(
                                go.Scatter(x=g.index, y=g.values,
                                           mode="lines+markers",
                                           line=dict(color="#f39c12", width=2),
                                           name="EC"), row=1, col=2)
                    base_layout(fig_mix)
                    fig_mix.update_xaxes(tickformat="%m-%d")
                    st.plotly_chart(fig_mix, width="stretch")
                else:
                    st.info("pH·EC 열이 없어서 pH/EC 그래프는 표시하지 않았어요.")

                col_t, col_h = st.columns(2)
                with col_t:
                    if t_fig:
                        st.plotly_chart(t_fig, width="stretch")
                    else:
                        st.info("온도 데이터가 충분하지 않아요.")
                with col_h:
                    if h_fig:
                        st.plotly_chart(h_fig, width="stretch")
                    else:
                        st.info("습도 데이터가 충분하지 않아요.")
            else:
                st.info("시간(time) 열이 없어서 시계열 그래프를 표시하지 못했어요.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 환경 데이터 원본 보기 + CSV 다운로드"):
                if env_df.empty:
                    st.info("데이터가 없습니다.")
                else:
                    display_df = env_df.copy()
                    if "time" in display_df.columns:
                        display_df["time"] = display_df["time"].dt.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(display_df, width="stretch", hide_index=True)
                    st.download_button(
                        label="📥 환경 CSV 다운로드",
                        data=download_csv(display_df),
                        file_name=f"{school_name}_환경데이터_다운로드.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===================================================================
#  TAB 3 — 생육 결과
# ===================================================================
with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 이 탭의 내용이 여기에 표시됩니다.")
        else:
            # ── 생중량 기준 정렬용 데이터 ──
            if "생중량" in growth_df.columns:
                bw_sorted = growth_df[["개체번호", "생중량"]].dropna().sort_values(
                    "생중량", ascending=False).reset_index(drop=True)
                n_be      = len(bw_sorted)
                top25     = int(np.ceil(n_be * 0.25))
                bot25     = int(np.ceil(n_be * 0.25))

            # ── 핵심 결과 카드 ──
            st.header("🥇 핵심 결과")
            kc1, kc2, kc3 = st.columns(3)
            bw_mean_v = safe_mean(growth_df["생중량"]) if "생중량" in growth_df.columns else None
            kc1.metric("평균 생중량 (g)",
                       f"{bw_mean_v:.2f}" if bw_mean_v is not None else "—")

            if "생중량" in growth_df.columns and not bw_sorted.empty:
                top_row = bw_sorted.iloc[0]
                bot_row = bw_sorted.iloc[-1]
                top_id  = (int(top_row["개체번호"]) if pd.notna(top_row["개체번호"])
                           and float(top_row["개체번호"]) == int(top_row["개체번호"])
                           else top_row["개체번호"])
                bot_id  = (int(bot_row["개체번호"]) if pd.notna(bot_row["개체번호"])
                           and float(bot_row["개체번호"]) == int(bot_row["개체번호"])
                           else bot_row["개체번호"])
                kc2.metric("생중량 최대 개체",
                           f"개체 {top_id}",
                           f"{top_row['생중량']:.2f} g")
                kc3.metric("생중량 최소 개체",
                           f"개체 {bot_id}",
                           f"{bot_row['생중량']:.2f} g")
            else:
                kc2.info("생중량 데이터가 없어서 최대·최소 개체를 표시할 수 없어요.")
                kc3.info("생중량 데이터가 없어서 최대·최소 개체를 표시할 수 없어요.")

            # ── 개체별 생육 막대 그래프 ──
            st.header("개체별 생육 비교 (막대 그래프)")
            bar_cols = cols_in(growth_df, ["잎 수", "지상부 길이", "지하부 길이", "생중량"])

            def bar_chart(col):
                if col not in growth_df.columns or growth_df[col].dropna().empty:
                    return None
                df_b = growth_df[["개체번호", col]].dropna().sort_values("개체번호").reset_index(drop=True)
                colors = ["#5b8def"] * len(df_b)
                if col == "생중량" and n_be >= 4:
                    top_idx_set = set(bw_sorted.iloc[:top25].index)
                    for bi, bidx in enumerate(df_b.index):
                        if bidx in top_idx_set:
                            colors[bi] = "#ff6b6b"
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_b["개체번호"].astype(str),
                    y=df_b[col],
                    marker=dict(color=colors),
                    name=col,
                    hovertemplate=f"{col}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>"
                ))
                base_layout(fig, title=f"개체별 {col}")
                fig.update_xaxes(title="개체번호", tickangle=45)
                fig.update_yaxes(title=col)
                return fig

            bar_items = [c for c in sel_items if c in bar_cols]
            if bar_items:
                nc = 2
                rn = (len(bar_items) + nc - 1) // nc
                for r in range(rn):
                    cols_r = st.columns(nc)
                    for cidx in range(nc):
                        idx = r * nc + cidx
                        with cols_r[cidx]:
                            if idx < len(bar_items):
                                bf = bar_chart(bar_items[idx])
                                if bf:
                                    st.plotly_chart(bf, width="stretch")
                                else:
                                    st.info("데이터가 충분하지 않아요.")
                            else:
                                st.empty()
            else:
                st.info("선택한 생육 측정 항목이 데이터에 없어서 막대 그래프를 표시할 수 없어요.")

            # ── 생중량 분포 (박스플롯) ──
            st.header("생중량 분포")
            if "생중량" in growth_df.columns and growth_df["생중량"].dropna().shape[0] >= 3:
                bw_v       = growth_df["생중량"].dropna()
                bw_mean_v2 = bw_v.mean()
                bw_std_v   = bw_v.std()
                cv_v       = (bw_std_v / bw_mean_v2 * 100) if bw_mean_v2 else None
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=bw_v,
                    name="생중량 (g)",
                    fillcolor="#5b8def",
                    marker=dict(color="#5b8def", size=5),
                    boxmean=True,
                    hovertemplate="생중량: %{{y:.2f}} g<extra></extra>"
                ))
                base_layout(fig_box, title="개체별 생중량 분포 (박스플롯)")
                fig_box.update_yaxes(title="생중량 (g)")
                st.plotly_chart(fig_box, width="stretch")
                st.caption(
                    f"📐 **요약:** 평균 {bw_mean_v2:.3f} g, "
                    f"표준편차 {bw_std_v:.3f} g, "
                    f"CV(변동계수) {cv_v:.1f}%  "
                    f"(CV = 표준편차 ÷ 평균 × 100 — 평균이 달라도 퍼짐 정도를 비교할 수 있는 지표예요)"
                )
            else:
                st.info("생중량 데이터가 3개 미만이라 분포 그래프를 표시할 수 없어요.")

            # ── 상관관계 분석 (산점도 + 추세선) ──
            st.header("상관관계 분석 (산점도 + 추세선)")
            if "생중량" in growth_df.columns:
                corr_pairs = []
                if "잎 수" in growth_df.columns:
                    corr_pairs.append(("잎 수", "#27ae60"))
                if "지상부 길이" in growth_df.columns:
                    corr_pairs.append(("지상부 길이", "#8e44ad"))

                if corr_pairs:
                    n_pair = len(corr_pairs)
                    fig_sch = make_subplots(rows=1, cols=n_pair,
                                            subplot_titles=[f"{xc} vs 생중량" for xc, _ in corr_pairs])
                    for i, (xcol, color) in enumerate(corr_pairs, 1):
                        x_s = growth_df[xcol].dropna()
                        y_s = growth_df.loc[x_s.index, "생중량"].dropna()
                        common = x_s.index.intersection(y_s.index)
                        if len(common) < 2:
                            continue
                        xr = growth_df.loc[common, xcol]
                        yr = growth_df.loc[common, "생중량"]
                        fig_sch.add_trace(go.Scatter(
                            x=xr, y=yr, mode="markers",
                            marker=dict(color=color, size=8, opacity=0.7),
                            name=f"{xcol} vs 생중량",
                            hovertemplate=f"{xcol}: %{{x}}<br>생중량: %{{y:.2f}}<extra></extra>"
                        ), row=1, col=i)
                        tl = trendline_trace(xr, yr, color)
                        if tl:
                            fig_sch.add_trace(tl, row=1, col=i)
                        r_v = safe_corr(xr, yr)
                        if r_v is not None:
                            fig_sch.add_annotation(
                                text=f"r = {r_v:.3f}",
                                xref=f"x{i}", yref=f"y{i}",
                                x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                                font=dict(size=11, color=color),
                                showarrow=False, row=1, col=i
                            )
                    base_layout(fig_sch)
                    st.plotly_chart(fig_sch, width="stretch")

                    for xcol, _ in corr_pairs:
                        r_v = safe_corr(growth_df[xcol], growth_df["생중량"])
                        if r_v is not None:
                            st.caption(
                                f"📌 **{xcol} vs 생중량:** 상관계수 r = {r_v:.3f}  "
                                f"(r이 +1에 가까우면 함께 증가, -1에 가까우면 반대로 움직이는 경향이 있어요)"
                            )
                else:
                    st.info("잎 수·지상부 길이 중 생중량과 비교할 열이 데이터에 없어요.")
            else:
                st.info("생중량 열이 없어서 상관관계 분석을 표시할 수 없어요.")

            # ── 상위 25% vs 하위 25% 비교 ──
            st.header("생중량 상위 25% vs 하위 25% 비교")
            if "생중량" in growth_df.columns and n_be >= 4:
                top_avg = bw_sorted.head(top25)["생중량"].mean()
                bot_avg = bw_sorted.tail(bot25)["생중량"].mean()
                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(
                    x=[f"상위 25%\n({top25}개체)", f"하위 25%\n({bot25}개체)"],
                    y=[top_avg, bot_avg],
                    marker=dict(color=["#27ae60", "#e74c3c"]),
                    hovertemplate="%{{x}}<br>평균 생중량: %{{y:.2f}} g<extra></extra>"
                ))
                base_layout(fig_q, title="상위 25% vs 하위 25% 평균 생중량 비교")
                fig_q.update_yaxes(title="평균 생중량 (g)")
                st.plotly_chart(fig_q, width="stretch")
                st.caption(
                    f"🏆 상위 25% 평균: **{top_avg:.2f} g**  |  "
                    f"🌱 하위 25% 평균: **{bot_avg:.2f} g**  |  "
                    f"차이: **{top_avg - bot_avg:.2f} g**"
                )
            else:
                st.info("생중량 데이터가 4개 미만이라 상위·하위 25% 비교를 할 수 없어요.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 생육 결과 데이터 원본 보기 + CSV 다운로드"):
                if growth_df.empty:
                    st.info("데이터가 없습니다.")
                else:
                    st.dataframe(growth_df, width="stretch", hide_index=True)
                    st.download_button(
                        label="📥 생육 CSV 다운로드",
                        data=download_csv(growth_df),
                        file_name=f"{school_name}_생육결과_다운로드.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

st.caption("---")
st.caption("🌱 극지식물 생육 대시보드  |  업로드한 CSV 기반으로 분석 내용이 자동 표시됩니다.")
