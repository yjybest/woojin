# main.py — polar-plant-dashboard
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
st.markdown("""\
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── 인코딩 이중 시도 (bytes 직접 읽기) ─────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

# ── 열 이름 표준화 + 숫자 변환 + 생중량 자동 합성 + 시간 파싱 ──
def std_cols(df):
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
    if not fname:
        return "우리 학교"
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

def safe_std(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.std() if len(v) >= 2 else None

def safe_corr(x, y):
    d2 = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d2) < 2:
        return None
    return d2["x"].corr(d2["y"])

def trendline_trace(x, y, color):
    d2 = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d2) < 2:
        return None
    sl, ic = np.polyfit(d2["x"], d2["y"], 1)
    xs = np.array([d2["x"].min(), d2["x"].max()])
    return go.Scatter(x=xs, y=sl * xs + ic, mode="lines",
                      line=dict(color=color, dash="dash", width=2),
                      name=f"추세선 (r={safe_corr(x, y):.3f})")

def download_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def base_layout(fig, title):
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                      title=title, margin=dict(l=40, r=20, t=44, b=40))
    return fig

def scatter_data(x, y):
    d2 = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d2) < 2:
        return None, None, None
    return d2["x"], d2["y"], d2["x"].corr(d2["y"])

def id_plain(v):
    return int(v) if pd.notna(v) and float(v) == int(v) else v

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="성범썜 사랑해요. 송도고 극지; 생육 결과", layout="wide")
st.title("🌱 송도고 극지; 생육 결과")

# ── 사이드바 ───────────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")
env_file    = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")
growth_sel  = st.sidebar.multiselect(
    "생육 측정 항목 선택 (그래프에 표시)",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"],
    default=["지상부 질량", "지하부 질량", "생중량"]
)
st.sidebar.divider()
st.sidebar.info("💡 CSV를 아직 업로드하지 않았다면 파일을 선택하세요. 선택한 탭만 표시됩니다.")

# ── 데이터 로드 (1회만 — 사이드바·탭 모두 재사용) ──────────────
with st.spinner("데이터를 읽고 정리 중이에요..."):
    env_df    = std_cols(read_csv(env_file))    if env_file    else pd.DataFrame()
    growth_df = std_cols(read_csv(growth_file)) if growth_file else pd.DataFrame()

school_name = school_name_from_filename(
    env_file.name if env_file else (growth_file.name if growth_file else "")
)
st.sidebar.caption(f"📌 감지된 학교: **{school_name}**")

tab1, tab2, tab3 = st.tabs([
    "🌱 상·하단 바이오매스 동반 성장 분석",
    "📏 키 중심의 가짜 성장과 진짜 생산성 규명",
    "🌡️ 고온 스트레스와 극단적 성장 양극화 분석",
])

# ===================================================================
#  TAB 1 — 상·하단 바이오매스 동반 성장 분석
# ===================================================================
with tab1:
    try:
        st.header("핵심 연구 질문: 지상부와 지하부는 함께 성장하는가?")

        st.subheader("📊 핵심 지표")
        c1, c2, c3, c4 = st.columns(4)
        am1 = safe_mean(growth_df["지상부 질량"]) if "지상부 질량" in growth_df.columns else None
        am2 = safe_mean(growth_df["지하부 질량"]) if "지하부 질량" in growth_df.columns else None
        c1.metric("평균 지상부 질량 (g)", f"{am1:.3f}" if am1 is not None else "—")
        c2.metric("평균 지하부 질량 (g)", f"{am2:.3f}" if am2 is not None else "—")
        bw_m = safe_mean(growth_df["생중량"]) if "생중량" in growth_df.columns else None
        c3.metric("평균 총 생중량 (g)", f"{bw_m:.3f}" if bw_m is not None else "—")
        if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
            r_val = safe_corr(growth_df["지상부 질량"], growth_df["지하부 질량"])
            r_col  = "normal" if r_val is None else \
                     ("green" if r_val >= 0.3 else "red" if r_val <= -0.3 else "normal")
            r_note = f"r = {r_val:.3f}" if r_val is not None else "—"
        else:
            r_val, r_note, r_col = None, "—", "normal"
        c4.metric("지상부·지하부 질량 상관계수", r_note, delta_color=r_col)

        st.caption(
            "📐 **상관계수 r**은 -1 ~ +1 사이 값이에요. +1에 가까우면 두 값이 함께 증가하는 "
            "경향, 0에 가까우면 서로 관계가 뚜렷하지 않다는 뜻이에요."
            "\n🌿 지상부·지하부 질량의 r이 **양수**면 지상부가 큰 개체가 지하부도 큰 편이라는 "
            "뜻으로, **상·하단 바이오매스가 동반 성장**한다는 증거가 됩니다."
        )

        st.subheader("📈 주요 그래프")

        # ① 지상부 질량 vs 지하부 질량 산점도 + 추세선
        st.markdown("### ① 지상부 질량 vs 지하부 질량 (산점도 + 추세선)")
        if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
            gx, gy, r_g = scatter_data(growth_df["지상부 질량"],
                                       growth_df["지하부 질량"])
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=gx, y=gy, mode="markers",
                                       marker=dict(color="#27ae60", size=8, opacity=0.75),
                                       name="개체별 (지상부 vs 지하부)",
                                       hovertemplate="지상부: %{{x:.3f}} g<br>"
                                                     "지하부: %{{y:.3f}} g<extra></extra>"))
            tl = trendline_trace(growth_df["지상부 질량"],
                                 growth_df["지하부 질량"], "#27ae60")
            if tl:
                fig_g.add_trace(tl)
            if r_g is not None:
                fig_g.add_annotation(text=f"r = {r_g:.3f}",
                                     xref="paper", yref="paper",
                                     x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                                     font=dict(size=12, color="#27ae60",
                                               weight="bold"),    # [FIX] bold → weight
                                     showarrow=False)
            fig_g.update_xaxes(title="지상부 질량 (g)")
            fig_g.update_yaxes(title="지하부 질량 (g)")
            base_layout(fig_g, title="지상부 질량 vs 지하부 질량 산점도")
            st.plotly_chart(fig_g, width="stretch")
        else:
            st.info("지상부 질량·지하부 질량 열이 모두 없어서 산점도를 표시할 수 없어요.")

        # ② 개체별 누적 막대그래프
        st.markdown("### ② 개체별 지상부·지하부 질량 비교 (누적 막대그래프)")
        if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
            grp = growth_df.groupby("개체번호", as_index=False)[
                ["지상부 질량", "지하부 질량"]].sum()
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(x=grp["개체번호"].astype(str),
                                   y=grp["지상부 질량"],
                                   name="지상부 질량",
                                   marker=dict(color="#2980b9"),
                                   hovertemplate="지상부: %{{y:.3f}} g<extra></extra>"))
            fig_s.add_trace(go.Bar(x=grp["개체번호"].astype(str),
                                   y=grp["지하부 질량"],
                                   name="지하부 질량",
                                   marker=dict(color="#27ae60"),
                                   hovertemplate="지하부: %{{y:.3f}} g<extra></extra>"))
            fig_s.update_layout(barmode="stack",
                                xaxis_title="개체번호", yaxis_title="질량 (g)",
                                bargap=0.15)
            base_layout(fig_s, title="개체별 지상부·지하부 질량 누적 비교")
            st.plotly_chart(fig_s, width="stretch")
        else:
            st.info("지상부 질량·지하부 질량 열이 없어서 누적 막대그래프를 표시할 수 없어요.")

        # ③ 개체별 구성비 그래프
        st.markdown("### ③ 개체별 지상부·지하부 구성비 (지상부 질량 비율)")
        if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
            grp2 = growth_df.groupby("개체번호", as_index=False)[
                ["지상부 질량", "지하부 질량"]].sum()
            total2 = grp2["지상부 질량"] + grp2["지하부 질량"]
            ratio = np.where(total2 > 0, grp2["지상부 질량"] / total2, 0.5)
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=grp2["개체번호"].astype(str),
                                   y=ratio,
                                   name="지상부 질량 비율",
                                   marker=dict(color=np.where(
                                       ratio >= 0.6, "#2980b9", "#27ae60")),
                                   hovertemplate="지상부 비율: %{{y:.1%}}<extra></extra>"))
            fig_r.add_hline(y=0.5, line_dash="dot", line_color="gray",
                            line_width=1.5,
                            annotation_text="지상부:지하부 = 1:1",
                            annotation_position="right")
            fig_r.update_yaxis(tickformat=".0%", range=[0, 1])
            base_layout(fig_r, title="개체별 지상부 질량 비율")
            st.plotly_chart(fig_r, width="stretch")
            st.caption(
                "📊 파란 막대(지상부 비율 ≥ 60%)는 **지상부 중심 성장**, 초록 막대(< 60%)는 "
                "**지하부 중심 성장** 개체를 나타내요.\n"
                "지상부 비율이 50%를 크게 웃도는 개체가 많으면 → 지상부 중심 성장 성향이 강해요."
            )
        else:
            st.info("지상부 질량·지하부 질량 열이 없어서 구성비 그래프를 표시할 수 없어요.")

        # ── 핵심 분석 ──
        st.subheader("🔍 핵심 분석")
        if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
            r2_val = safe_corr(growth_df["지상부 질량"], growth_df["지하부 질량"])
            grp3 = growth_df.groupby("개체번호", as_index=False)[
                ["지상부 질량", "지하부 질량"]].sum()
            grp3["총질량"] = grp3["지상부 질량"] + grp3["지하부 질량"]
            grp3["지상부비율"] = np.where(
                grp3["총질량"] > 0, grp3["지상부 질량"] / grp3["총질량"], 0.5)
            above_cnt  = int((grp3["지상부비율"] >= 0.6).sum())
            below_cnt  = int((grp3["지상부비율"] <= 0.4).sum())
            middle_cnt = len(grp3) - above_cnt - below_cnt
            if r2_val is not None:
                if r2_val > 0.3:
                    direction = "동반 성장 경향이 뚜렷해요"
                elif r2_val < -0.3:
                    direction = "서로 반대 방향으로 움직이는 경향이 보여요"
                else:
                    direction = "뚜렷한 관계라고 보기 어려워요"
            else:
                direction = "데이터가 충분하지 않아요"
            st.markdown(f"""
**① 상·하단 바이오매스의 동반 성장 여부**
- 지상부·지하부 질량 상관계수: **r = {r2_val:.3f}**  ({direction})
- 지상부 중심 성장 개체: **{above_cnt} 개체** (지상부 비율 ≥ 60%)
- 지하부 중심 성장 개체: **{below_cnt} 개체** (지상부 비율 ≤ 40%)
- 균형 성장 개체: **{middle_cnt} 개체**

**② 지상부 중심 vs 지하부 중심 성장 개체 구분**
- 지상부 비율이 높을수록 잎·줄기 성장에 에너지를 많이 쓴 개체예요.
- 지하부 비율이 높을수록 뿌리 발달에 에너지를 많이 쓴 개체예요.

**③ 바이오매스 배분 특성**
- {school_name}의 평균 지상부 비율: **{grp3["지상부비율"].mean():.1%}**
- 지상부·지하부 질량이 함께 증가하는지 확인하면 식물이 위·아래로 균형 있게 자랐는지 알 수 있어요.
            """.strip())
        else:
            st.info("지상부 질량·지하부 질량 열이 없어서 핵심 분석을 표시할 수 없어요.")

        with st.expander("📄 생육 결과 데이터 원본 보기 + CSV 다운로드"):
            if growth_df.empty:
                st.info("데이터가 없습니다.")
            else:
                st.dataframe(growth_df, width="stretch", hide_index=True)
                st.download_button("📥 생육 CSV 다운로드",
                                   download_csv(growth_df),
                                   file_name=f"{school_name}_생육결과_다운로드.csv",
                                   mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===================================================================
#  TAB 2 — 키 중심의 가짜 성장과 진짜 생산성 규명
# ===================================================================
with tab2:
    try:
        st.header("핵심 연구 질문: 키가 큰 식물이 정말 생산성이 높은 식물인가?")

        st.subheader("📊 핵심 지표")
        c1, c2, c3, c4 = st.columns(4)
        sl_m = safe_mean(growth_df["지상부 길이"]) if "지상부 길이" in growth_df.columns else None
        c1.metric("평균 지상부 길이 (mm)", f"{sl_m:.1f}" if sl_m is not None else "—")
        c2.metric("평균 생중량 (g)", f"{bw_m:.3f}" if bw_m is not None else "—")
        if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
            r_sl = safe_corr(growth_df["지상부 길이"], growth_df["생중량"])
            r_sl_col = ("green" if r_sl is not None and r_sl >= 0.3
                        else "red" if r_sl is not None and r_sl <= -0.3
                        else "normal")
            r_sl_note = f"r = {r_sl:.3f}" if r_sl is not None else "—"
        else:
            r_sl, r_sl_note, r_sl_col = None, "—", "normal"
        c3.metric("키와 생중량 상관계수", r_sl_note, delta_color=r_sl_col)
        if "생중량" in growth_df.columns and "지상부 길이" in growth_df.columns:
            eff_vals = (growth_df["생중량"] / growth_df["지상부 길이"]).dropna()
            eff_m = safe_mean(eff_vals) if len(eff_vals) >= 1 else None
        else:
            eff_m = None
        c4.metric("평균 생물량 효율 (g/mm)", f"{eff_m:.4f}" if eff_m is not None else "—")

        st.caption(
            "📐 **상관계수 r**이 +1에 가까우면 지상부 길이가 길수록 생중량도 큰 경향, "
            "0에 가까우면 키만으로는 생산성을 예측하기 어렵다는 뜻이에요."
            "\n🔬 **생물량 효율(g/mm)** = 생중량(g) ÷ 지상부 길이(mm) — 같은 키라도 "
            "얼마나 많이 생물량을 만들었는지 나타내는 생산성 지표예요. 값이 클수록 "
            "**키는 같아도 더 알차게 자란** 개체예요."
        )

        st.subheader("📈 주요 그래프")

        # ① 지상부 길이 vs 생중량 산점도
        st.markdown("### ① 지상부 길이 vs 생중량 (산점도 + 추세선)")
        if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
            sx, sy, r_sx = scatter_data(growth_df["지상부 길이"],
                                        growth_df["생중량"])
            fig_sx = go.Figure()
            fig_sx.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                                        marker=dict(color="#8e44ad", size=8, opacity=0.75),
                                        name="개체별 (키 vs 생중량)",
                                        hovertemplate="지상부 길이: %{{x}} mm<br>"
                                                      "생중량: %{{y:.3f}} g<extra></extra>"))
            tl_sx = trendline_trace(growth_df["지상부 길이"],
                                    growth_df["생중량"], "#8e44ad")
            if tl_sx:
                fig_sx.add_trace(tl_sx)
            if r_sx is not None:
                fig_sx.add_annotation(text=f"r = {r_sx:.3f}",
                                      xref="paper", yref="paper",
                                      x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                                      font=dict(size=12, color="#8e44ad",
                                                weight="bold"),   # [FIX] bold → weight
                                      showarrow=False)
            fig_sx.update_xaxes(title="지상부 길이 (mm)")
            fig_sx.update_yaxes(title="생중량 (g)")
            base_layout(fig_sx, title="지상부 길이 vs 생중량 산점도")
            st.plotly_chart(fig_sx, width="stretch")
            if r_sx is not None and r_sx > 0:
                cap = ("r이 양수면 키가 클수록 생중량이 큰 경향 → 길이 성장과 생물량 생산이 "
                       "대체로 함께 움직여요.")
            elif r_sx is not None:
                cap = ("r이 낮거나 음수면 키가 크다고 생중량이 반드시 큰 것은 아니에요. "
                       "여기서 **가짜 성장**(키만 크고 생물량은 적은 개체)이 나타날 수 있어요.")
            else:
                cap = ""
            st.caption(cap)

        # ② 잎 수 vs 생중량 산점도
        st.markdown("### ② 잎 수 vs 생중량 (산점도 + 추세선)")
        if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
            lx, ly, r_lx = scatter_data(growth_df["잎 수"],
                                         growth_df["생중량"])
            fig_lx = go.Figure()
            fig_lx.add_trace(go.Scatter(x=lx, y=ly, mode="markers",
                                        marker=dict(color="#27ae60", size=8, opacity=0.75),
                                        name="개체별 (잎 수 vs 생중량)",
                                        hovertemplate="잎 수: %{{x}} 장<br>"
                                                      "생중량: %{{y:.3f}} g<extra></extra>"))
            tl_lx = trendline_trace(growth_df["잎 수"], growth_df["생중량"], "#27ae60")
            if tl_lx:
                fig_lx.add_trace(tl_lx)
            if r_lx is not None:
                fig_lx.add_annotation(text=f"r = {r_lx:.3f}",
                                      xref="paper", yref="paper",
                                      x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                                      font=dict(size=12, color="#27ae60",
                                                weight="bold"),   # [FIX] bold → weight
                                      showarrow=False)
            fig_lx.update_xaxes(title="잎 수 (장)")
            fig_lx.update_yaxes(title="생중량 (g)")
            base_layout(fig_lx, title="잎 수 vs 생중량 산점도")
            st.plotly_chart(fig_lx, width="stretch")
        else:
            st.info("잎 수 또는 생중량 열이 없어서 산점도를 표시할 수 없어요.")

        # ③ 개체별 키 순위 vs 생중량 순위 비교
        st.markdown("### ③ 개체별 키 순위 vs 생중량 순위 비교")
        if {"지상부 길이", "생중량"} <= set(growth_df.columns):
            df_r = growth_df[["개체번호", "지상부 길이", "생중량"]].dropna().copy()
            if len(df_r) >= 3:
                df_r["길이순위"] = df_r["지상부 길이"].rank(
                    ascending=False, method="min")
                df_r["중량순위"] = df_r["생중량"].rank(
                    ascending=False, method="min")
                df_r["순위차이"] = df_r["길이순위"] - df_r["중량순위"]
                df_r = df_r.sort_values("순위차이").reset_index(drop=True)
                fig_rr = go.Figure()
                fig_rr.add_trace(go.Bar(x=df_r["개체번호"].astype(str),
                                        y=df_r["지상부 길이"],
                                        name="지상부 길이 (mm)",
                                        marker=dict(color="#8e44ad"),
                                        hovertemplate="키 순위: %{{customdata[0]}}위<br>"
                                                      "길이: %{{y}} mm<extra></extra>",
                                        customdata=df_r["길이순위"].values[:, None]))
                fig_rr.add_trace(go.Bar(x=df_r["개체번호"].astype(str),
                                        y=df_r["생중량"],
                                        name="생중량 (g)",
                                        marker=dict(color="#27ae60"),
                                        hovertemplate="생중량 순위: %{{customdata[0]}}위<br>"
                                                      "생중량: %{{y:.3f}} g<extra></extra>",
                                        customdata=df_r["중량순위"].values[:, None]))
                fig_rr.update_layout(barmode="group",
                                     xaxis_title="개체번호",
                                     yaxis_title="값",
                                     bargap=0.25, bargroupgap=0.15)
                base_layout(fig_rr, title="개체별 키 순위 vs 생중량 순위 비교")
                st.plotly_chart(fig_rr, width="stretch")
                avg_gap = int(df_r["순위차이"].abs().mean()) if len(df_r) > 0 else None
                st.caption(
                    f"📊 각 개체의 **키 순위**와 **생중량 순위**를 나란히 비교했어요. "
                    f"두 순위가 크게 다른 개체일수록 키만으로는 생산성을 예측하기 어려운 개체예요. "
                    f"평균 순위 차이: 약 **{avg_gap}순위** "
                    f"(0에 가까울수록 키와 생중량이 비슷한 순위로 정렬됨)"
                )
            else:
                st.info("데이터가 3개체 미만이라 순위 비교를 표시할 수 없어요.")
        else:
            st.info("지상부 길이 또는 생중량 열이 없어서 순위 비교를 표시할 수 없어요.")

        # ④ 개체별 생물량 효율 순위 막대그래프
        st.markdown("### ④ 개체별 생물량 효율 순위 (g/mm)")
        if "생중량" in growth_df.columns and "지상부 길이" in growth_df.columns:
            df_e = growth_df[["개체번호", "생중량", "지상부 길이"]].dropna().copy()
            df_e["생물량효율"] = df_e["생중량"] / df_e["지상부 길이"]
            df_e = df_e[df_e["지상부 길이"] > 0].sort_values(
                "생물량효율", ascending=False).reset_index(drop=True)
            if len(df_e) >= 1:
                colors_e = ["#27ae60" if i < len(df_e) * 0.25 else "#8e44ad"
                            for i in range(len(df_e))]
                fig_e = go.Figure()
                fig_e.add_trace(go.Bar(x=df_e["개체번호"].astype(str),
                                       y=df_e["생물량효율"],
                                       marker=dict(color=colors_e),
                                       name="생물량 효율 (g/mm)",
                                       hovertemplate="효율: %{{y:.4f}} g/mm<extra></extra>"))
                base_layout(fig_e, title="개체별 생물량 효율 순위 (초록 = 상위 25%)")
                fig_e.update_xaxes(title="개체번호", tickangle=45)
                fig_e.update_yaxes(title="생물량 효율 (g/mm)")
                st.plotly_chart(fig_e, width="stretch")
            else:
                st.info("유효한 효율 값이 하나도 없어요.")
        else:
            st.info("생중량 또는 지상부 길이 열이 없어서 효율 순위를 표시할 수 없어요.")

        # ⑤ 상관관계 히트맵
        st.markdown("### ⑤ 생육 특성 상관관계 히트맵")
        num_cols = cols_in(growth_df,
                           ["잎 수", "지상부 길이", "지하부 길이",
                            "지상부 질량", "지하부 질량", "생중량"])
        if len(num_cols) >= 2:
            corr_mat = growth_df[num_cols].corr()
            fig_hm = px.imshow(corr_mat,
                               text_auto=".2f",
                               color_continuous_scale="RdBu_r",
                               zmin=-1, zmax=1,
                               labels=dict(x="측정 항목", y="측정 항목",
                                           color="상관계수 r"))
            fig_hm.update_xaxes(side="top")
            base_layout(fig_hm, title="생육 특성 간 상관관계 히트맵")
            fig_hm.update_layout(coloraxis_colorbar=dict(title="r"), width=800)
            st.plotly_chart(fig_hm, width="stretch")
            st.caption(
                "📊 색이 **빨강(r ≈ +1)**에 가까우면 두 항목이 함께 증가하는 경향, "
                "**파랑(r ≈ -1)**이면 반대로 움직이는 경향이에요. "
                "**흰색(r ≈ 0)**은 관계가 뚜렷하지 않다는 뜻이에요."
            )
        else:
            st.info("상관계수를 계산할 열이 2개 이상 필요해서 히트맵을 표시할 수 없어요.")

        # ── 생산성 지표 & 핵심 분석 ──
        st.subheader("🔬 생산성 지표 & 핵심 분석")
        if "생중량" in growth_df.columns and "지상부 길이" in growth_df.columns:
            df_a = growth_df[["개체번호", "생중량", "지상부 길이"]].dropna().copy()
            mask = df_a["지상부 길이"] > 0
            df_a = df_a[mask].copy()
            df_a["효율"] = df_a["생중량"] / df_a["지상부 길이"]
            df_a_k = df_a.sort_values("지상부 길이", ascending=False).reset_index(drop=True)
            df_a_e = df_a.sort_values("효율", ascending=False).reset_index(drop=True)
            tall_n    = min(5, len(df_a_k))
            high_eff_n = min(5, len(df_a_e))
            low_eff_n  = min(5, len(df_a_e))
            tall      = df_a_k.head(tall_n)
            high_eff  = df_a_e.head(high_eff_n)
            low_eff   = df_a_e.tail(low_eff_n).iloc[::-1]
            median_bw = df_a["생중량"].median()

            fake_lines = []
            for _, r in tall.iterrows():
                if r["생중량"] < median_bw:
                    fake_lines.append(
                        f"• 개체 {id_plain(r['개체번호'])}: 키 {int(r['지상부 길이'])} mm, "
                        f"생중량 {r['생중량']:.3f} g, 효율 {r['효율']:.4f} g/mm"
                    )
            high_lines = []
            for _, r in high_eff.iterrows():
                high_lines.append(
                    f"• 개체 {id_plain(r['개체번호'])}: 키 {int(r['지상부 길이'])} mm, "
                    f"생중량 {r['생중량']:.3f} g, 효율 {r['효율']:.4f} g/mm"
                )

            st.markdown(f"""
**키 중심 성장 vs 진짜 생산성**

| 구분 | 정의 | 개수 |
|------|------|------|
| 키 큰 개체 (상위 {tall_n}) | 지상부 길이가 긴 순서로 상위 | {tall_n} 개체 |
| 고효율 개체 (상위 {high_eff_n}) | 생물량 효율(g/mm) 상위 | {high_eff_n} 개체 |
| 저효율 개체 (하위 {low_eff_n}) | 생물량 효율이 낮은 개체 | {low_eff_n} 개체 |

**키가 크지만 생중량이 낮은 개체** (가짜 성장 의심):
{"".join(fake_lines) if fake_lines else "없음 (키 상위 개체 모두 생중량도 상위권)"}

**키에 비해 생물량이 높은 개체** (진짜 생산성 우수):
{"".join(high_lines) if high_lines else "없음"}

**단순 길이 성장과 실제 생물량 생산의 차이**
- 지상부 길이는 줄기가 얼마나 길어졌는지를, 생중량은 실제 만들어진 생물량(잎+줄기+뿌리 무게)을 나타내요.
- 키 순위 상위권 + 생중량 순위 상위 → **진짜 생산성 우수**
- 키 순위 상위권 + 생중량 순위 하위 → **가짜 성장 의심** (길이만 늘어나고 생물량은 적음)
- 생물량 효율이 높은 개체일수록 제한된 길이로도 많은 생물량을 생산한 **고효율 개체**예요.
            """.strip())
        else:
            st.info("생중량 또는 지상부 길이 열이 없어서 생산성 분석을 표시할 수 없어요.")

        with st.expander("📄 생육 결과 데이터 원본 보기 + CSV 다운로드"):
            if growth_df.empty:
                st.info("데이터가 없습니다.")
            else:
                st.dataframe(growth_df, width="stretch", hide_index=True)
                st.download_button("📥 생육 CSV 다운로드",
                                   download_csv(growth_df),
                                   file_name=f"{school_name}_생육결과_다운로드.csv",
                                   mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===================================================================
#  TAB 3 — 고온 스트레스와 극단적 성장 양극화 분석
# ===================================================================
with tab3:
    try:
        st.header("핵심 연구 질문: 고온과 미세 환경은 개체별 성장 격차와 어떤 관계가 있는가?")

        st.subheader("📊 핵심 지표")
        c1, c2, c3, c4 = st.columns(4)
        t_vals = env_df["temperature"].dropna() if "temperature" in env_df.columns else pd.Series(dtype=float)
        h_vals = env_df["humidity"].dropna()    if "humidity"    in env_df.columns else pd.Series(dtype=float)

        if t_vals.any():
            c1.metric("평균 온도 (℃)", f"{safe_mean(t_vals):.1f}",
                      delta=f"최고 {safe_max(t_vals):.1f}℃")
            c2.metric("최고 온도 (℃)", f"{safe_max(t_vals):.1f}")
        else:
            c1.info("온도 열이 없어요.")
            c2.info("온도 열이 없어요.")
        if h_vals.any():
            c3.metric("평균 습도 (%)", f"{safe_mean(h_vals):.1f}")
        else:
            c3.info("습도 열이 없어요.")
        if "생중량" in growth_df.columns:
            bw_vals = growth_df["생중량"].dropna()
            sd_val  = safe_std(bw_vals)
            bw_mean_v = safe_mean(bw_vals)
            if sd_val is not None and bw_mean_v:
                cv_note = f"CV {sd_val/bw_mean_v*100:.1f}%"
            else:
                cv_note = None
            c4.metric("생중량 표준편차 (g)",
                      f"{sd_val:.3f}" if sd_val is not None else "—",
                      delta=cv_note)
        else:
            c4.info("생중량 열이 없어요.")

        st.caption(
            "📐 **표준편차(SD)**는 값이 평균에서 얼마나 떨어져 있는지 나타내는 숫자예요. "
            "값이 클수록 개체 간 생육 차이가 크다는 뜻이에요."
            "\n📊 **CV(변동계수)** = 표준편차 ÷ 평균 × 100 — 평균이 달라도 개체 간 퍼짐 정도를 비교할 수 있어요."
        )

        st.subheader("📈 주요 그래프")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        # ① 시간별 온도 변화
        with r1c1:
            st.markdown("### ① 시간별 온도 변화")
            if "time" in env_df.columns and "temperature" in env_df.columns:
                ts = env_df[["time", "temperature"]].dropna()
                if not ts.empty:
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=ts["time"], y=ts["temperature"],
                                                mode="lines",
                                                line=dict(color="#ff6b6b", width=1.5),
                                                name="온도 (°C)"))
                    fig_t.add_hline(y=safe_mean(ts["temperature"]),
                                    line_dash="dot", line_color="gray",
                                    annotation_text=f"평균 {safe_mean(ts['temperature']):.1f}℃",
                                    annotation_position="right")
                    base_layout(fig_t, title="시간별 온도 변화")
                    fig_t.update_xaxes(tickformat="%m-%d %H:%M", tickangle=45)
                    fig_t.update_yaxes(title="온도 (℃)")
                    st.plotly_chart(fig_t, width="stretch")
                else:
                    st.info("온도 데이터가 없어요.")
            else:
                st.info("time 또는 temperature 열이 없어요.")

        # ② 시간별 습도 변화
        with r1c2:
            st.markdown("### ② 시간별 습도 변화")
            if "time" in env_df.columns and "humidity" in env_df.columns:
                hs = env_df[["time", "humidity"]].dropna()
                if not hs.empty:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=hs["time"], y=hs["humidity"],
                                                mode="lines",
                                                line=dict(color="#4ecdc4", width=1.5),
                                                name="습도 (%)"))
                    fig_h.add_hline(y=safe_mean(hs["humidity"]),
                                    line_dash="dot", line_color="gray",
                                    annotation_text=f"평균 {safe_mean(hs['humidity']):.1f}%",
                                    annotation_position="right")
                    base_layout(fig_h, title="시간별 습도 변화")
                    fig_h.update_xaxes(tickformat="%m-%d %H:%M", tickangle=45)
                    fig_h.update_yaxes(title="습도 (%)")
                    st.plotly_chart(fig_h, width="stretch")
                else:
                    st.info("습도 데이터가 없어요.")
            else:
                st.info("time 또는 humidity 열이 없어요.")

        # ③ 온도 분포 히스토그램 + KDE
        with r2c1:
            st.markdown("### ③ 온도 분포 히스토그램")
            if "temperature" in env_df.columns:
                tv2 = env_df["temperature"].dropna()
                if len(tv2) >= 5:
                    fig_th = go.Figure()
                    fig_th.add_trace(go.Histogram(x=tv2, nbinsx=30,
                                                   marker=dict(color="#ff6b6b", opacity=0.65),
                                                   name="측정 횟수",
                                                   hovertemplate="온도: %{{x:.1f}}℃<br>횟수: %{{y}}<extra></extra>"))
                    if len(tv2) > 1:
                        xs_k = np.linspace(tv2.min(), tv2.max(), 200)
                        kx, kstd = tv2.mean(), tv2.std()
                        if kstd > 0:
                            ys_k = (1 / (kstd * np.sqrt(2 * np.pi)) *
                                    np.exp(-0.5 * ((xs_k - kx) / kstd) ** 2))
                            ys_k *= len(tv2) * (tv2.max() - tv2.min()) / 30
                            fig_th.add_trace(go.Scatter(x=xs_k, y=ys_k, mode="lines",
                                                        line=dict(color="#c0392b", width=2),
                                                        name="분포 곡선"))
                    fig_th.add_vline(x=safe_mean(tv2), line_dash="dot",
                                     line_color="gray",
                                     annotation_text=f"평균 {safe_mean(tv2):.1f}℃",
                                     annotation_position="top")
                    base_layout(fig_th, title="온도 분포 (히스토그램 + 분포 곡선)")
                    fig_th.update_xaxes(title="온도 (℃)")
                    fig_th.update_yaxes(title="측정 횟수")
                    st.plotly_chart(fig_th, width="stretch")
                else:
                    st.info("온도 데이터가 충분하지 않아요.")
            else:
                st.info("temperature 열이 없어요.")

        # ④ 온도 vs 습도 산점도
        with r2c2:
            st.markdown("### ④ 온도 vs 습도 산점도")
            if {"temperature", "humidity"} <= set(env_df.columns):
                ec2 = env_df[["temperature", "humidity"]].dropna()
                if len(ec2) >= 5:
                    fig_ths = go.Figure()
                    fig_ths.add_trace(go.Scatter(x=ec2["temperature"],
                                                 y=ec2["humidity"],
                                                 mode="markers",
                                                 marker=dict(color="#4ecdc4", size=4, opacity=0.5),
                                                 name="측정값",
                                                 hovertemplate="온도: %{{x:.1f}}℃<br>"
                                                               "습도: %{{y:.1f}}%<extra></extra>"))
                    r_th = safe_corr(ec2["temperature"], ec2["humidity"])
                    tl_th = trendline_trace(ec2["temperature"], ec2["humidity"], "#2c3e50")
                    if tl_th:
                        fig_ths.add_trace(tl_th)
                    if r_th is not None:
                        fig_ths.add_annotation(text=f"r = {r_th:.3f}",
                                               xref="paper", yref="paper",
                                               x=0.95, y=0.05, xanchor="right",
                                               yanchor="bottom",
                                               font=dict(size=12, color="#2c3e50",
                                                         weight="bold"),    # [FIX] bold → weight
                                               showarrow=False)
                    base_layout(fig_ths, title="온도 vs 습도 산점도")
                    fig_ths.update_xaxes(title="온도 (℃)")
                    fig_ths.update_yaxes(title="습도 (%)")
                    st.plotly_chart(fig_ths, width="stretch")
                else:
                    st.info("데이터가 충분하지 않아요.")
            else:
                st.info("temperature 또는 humidity 열이 없어요.")

        # ── 생중량 상위 5 vs 하위 5 비교 ──
        st.subheader("🏆 생중량 상위 5개체 vs 하위 5개체 비교")
        if "생중량" in growth_df.columns:
            bw_sorted = growth_df[["개체번호", "생중량"]].dropna().sort_values(
                "생중량", ascending=False).reset_index(drop=True)
            n_bw = len(bw_sorted)
            if n_bw >= 2:
                top5 = bw_sorted.head(5)
                bot5 = bw_sorted.tail(5).iloc[::-1]
                t_ids  = [id_plain(r["개체번호"]) for _, r in top5.iterrows()]
                b_ids  = [id_plain(r["개체번호"]) for _, r in bot5.iterrows()]
                t_vals = top5["생중량"].tolist()
                b_vals = bot5["생중량"].tolist()
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown("#### 🥇 생중량 상위 5개체")
                    for i, (id_, v) in enumerate(zip(t_ids, t_vals), 1):
                        st.markdown(f"**{i}위** · 개체 {id_} · **{v:.3f} g**")
                with k2:
                    st.markdown("#### 🌱 생중량 하위 5개체")
                    for i, (id_, v) in enumerate(zip(b_ids, b_vals), 1):
                        st.markdown(f"**{i}위** · 개체 {id_} · **{v:.3f} g**")

                top_mean = np.mean(t_vals)
                bot_mean = np.mean(b_vals)
                fig_topbot = go.Figure()
                fig_topbot.add_trace(go.Bar(x=["상위 5개체 평균", "하위 5개체 평균"],
                                            y=[top_mean, bot_mean],
                                            marker=dict(color=["#27ae60", "#e74c3c"]),
                                            hovertemplate="%{{x}}<br>평균 생중량: %{{y:.3f}} g<extra></extra>"))
                base_layout(fig_topbot, title="상위 5 vs 하위 5 평균 생중량")
                fig_topbot.update_yaxes(title="생중량 (g)")
                st.plotly_chart(fig_topbot, width="stretch")
                st.caption(
                    f"🏆 상위 5개체 평균: **{top_mean:.3f} g**  |  "
                    f"🌱 하위 5개체 평균: **{bot_mean:.3f} g**  |  "
                    f"차이: **{top_mean - bot_mean:.3f} g**"
                )
            else:
                st.info("생중량 데이터가 2개체 미만이라 비교할 수 없어요.")
        else:
            st.info("생중량 열이 없어서 상위·하위 5개체 비교를 표시할 수 없어요.")

        # ── 생육 특성별 성장 양극화 비교 ──
        st.subheader("📊 생육 특성별 성장 양극화 비교 그래프")
        spec_cols = cols_in(growth_df,
                            ["지상부 길이", "지하부 길이", "잎 수",
                             "지상부 질량", "지하부 질량", "생중량"])
        if len(spec_cols) >= 2:
            df_polar = growth_df[spec_cols].dropna()
            medians = df_polar.median()
            st.markdown("#### 생육 특성별 중앙값 및 분포 비교")
            fig_pol = go.Figure()
            colors_map = {
                "지상부 길이": "#8e44ad", "지하부 길이": "#16a085",
                "잎 수": "#27ae60", "지상부 질량": "#2980b9",
                "지하부 질량": "#27ae60", "생중량": "#2c3e50",
            }
            for col in spec_cols:
                vals = df_polar[col].dropna()
                if len(vals) < 1:
                    continue
                c = colors_map.get(col, "#333333")
                fig_pol.add_trace(go.Box(y=vals,
                                          name=f"{col}\n(중앙값 {medians[col]:.2f})",
                                          fillcolor=c,
                                          marker=dict(color=c, size=4),
                                          boxpoints="outliers",
                                          hovertemplate=f"{col}<br>값: %{{y:.3f}}<extra></extra>"))
            base_layout(fig_pol, title="생육 특성별 성장 양극화 비교 (박스플롯)")
            fig_pol.update_yaxes(title="측정값")
            fig_pol.update_xaxes(tickangle=45)
            st.plotly_chart(fig_pol, width="stretch")
            st.caption(
                "📊 각 측정 항목(잎 수, 지상부 길이 등)의 박스플롯을 나란히 비교했어요. "
                "박스가 넓게 퍼질수록 해당 항목에서 개체 간 차이가 크다는 뜻이에요."
                "\n⚠️ 개체별 환경 측정값이 없으므로 특정 개체가 고온 스트레스를 더 받았다고 단정하지 않고, "
                "전체 환경 데이터와 개체별 생육 차이의 연관성만 분석합니다."
            )
        else:
            st.info("비교할 생육 측정 항목이 2개 이상 필요해서 양극화 비교 그래프를 표시할 수 없어요.")

        # ── 핵심 분석 ──
        st.subheader("🔍 핵심 분석")
        if "temperature" in env_df.columns and t_vals.any():
            hot_thresh  = safe_mean(t_vals) + safe_std(t_vals) * 1.5 if (safe_mean(t_vals) is not None and safe_std(t_vals) is not None) else None
            hot_count   = int((t_vals > hot_thresh).sum()) if hot_thresh is not None else None
            hot_ratio   = hot_count / len(t_vals) * 100 if hot_count is not None else None
            hot_info = (
                f"\n- 고온 기준 (평균+1.5σ): **{hot_thresh:.1f}℃**"
                f"\n- 고온 측정 횟수: **{hot_count}회** ({hot_ratio:.1f}%)"
            ) if hot_thresh is not None else ""
        else:
            hot_thresh = hot_count = hot_ratio = None
            hot_info = "\n- 환경 데이터가 없어서 고온 분석은 생략해요."

        if "생중량" in growth_df.columns and bw_vals.any():
            bw_sd_v = safe_std(bw_vals)
            bw_mean_v2 = safe_mean(bw_vals)
            cv_v2 = bw_sd_v / bw_mean_v2 * 100 if (bw_sd_v is not None and bw_mean_v2) else None
            pol_info = (
                f"\n- 생중량 평균: **{bw_mean_v2:.3f} g**"
                f"\n- 생중량 표준편차: **{bw_sd_v:.3f} g**"
                f"\n- 변동계수(CV): **{cv_v2:.1f}%**"
                f"\n- CV가 클수록 개체 간 성장 격차가 크다는 뜻이에요."
            ) if bw_mean_v2 is not None else ""
        else:
            bw_sd_v = bw_mean_v2 = cv_v2 = None
            pol_info = "\n- 생육 데이터가 충분하지 않아요."

        st.markdown(f"""
**고온 발생 시기와 지속 정도**
- {school_name}의 평균 온도: **{safe_mean(t_vals):.1f}℃** (제공된 경우)
- 측정 기간 중 최고 온도: **{safe_max(t_vals):.1f}℃** (제공된 경우)
{hot_info}

**환경 데이터의 변동성과 생육 편차 비교**
{pol_info}

**상위·하위 개체의 생육 특성 차이**
- 첨부 CSV에 개체별 환경 측정값이 없으므로, 특정 개체가 고온 스트레스를 더 받았다고 단정하지 않아요.
- 대신 **전체 환경 조건**(평균 온도, 습도 변동 등)과 **개체 간 생육 차이**(생중량 CV, 표준편차)를 함께 살펴,
  환경 스트레스가 개체별 성장 양극화와 관련될 가능성을 탐색합니다.
- 고온 발생 비율이 높은 시기와 생육 데이터를 동시에 확인할 수 있다면,
  고온 기간 이후 생중량 편차가 커졌는지 비교해 볼 수 있어요.

**환경 스트레스와 성장 양극화의 연관성 탐색**
- 극지식물은 일반적으로 서늘하고 건조한 환경에 적응한 식물이므로,
  {school_name} 측정 기간 중 최고 온도가 **{safe_max(t_vals):.1f}℃**까지 올랐다면
  일부 개체는 고온 스트레스를 받았을 가능성이 있어요.
- 하지만 개체마다 미세 환경(빛, 통풍, 토양 등)이 다르기 때문에
  동일 환경에서도 성장 차이가 나타날 수 있고, 환경 데이터만으로는 개체별 원인을 특정할 수 없어요.
- 이 대시보드에서는 환경 데이터(시간별 온도·습도)와 생육 데이터(개체별 생중량 등)를
  같은 화면에서 비교해 보는 방식으로 연관성을 탐색하도록 구성했어요.
        """.strip())

        with st.expander("📄 환경 데이터 원본 보기 + CSV 다운로드"):
            if env_df.empty:
                st.info("데이터가 없습니다.")
            else:
                display_env = env_df.copy()
                if "time" in display_env.columns:
                    display_env["time"] = display_env["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(display_env, width="stretch", hide_index=True)
                st.download_button("📥 환경 CSV 다운로드",
                                   download_csv(display_env),
                                   file_name=f"{school_name}_환경데이터_다운로드.csv",
                                   mime="text/csv")

        with st.expander("📄 생육 결과 데이터 원본 보기 + CSV 다운로드"):
            if growth_df.empty:
                st.info("데이터가 없습니다.")
            else:
                st.dataframe(growth_df, width="stretch", hide_index=True)
                st.download_button("📥 생육 CSV 다운로드",
                                   download_csv(growth_df),
                                   file_name=f"{school_name}_생육결과_다운로드.csv",
                                   mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

st.caption("---")
st.caption("🌱 극지식물 생육 대시보드  |  업로드한 CSV 기반으로 분석 내용이 자동 표시됩니다.")
