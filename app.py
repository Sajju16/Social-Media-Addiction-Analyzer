from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from social_analyzer import (
    RISK_LABELS,
    Features,
    class_from_proba,
    predict_risk_proba,
    score_addiction,
    score_to_class,
    train_risk_model,
    extract_platform_breakdown,
    try_extract_time_series,
)


st.set_page_config(
    page_title="Social Media Addiction Analyzer",
    page_icon="SM",
    layout="wide",
)


def _card_css() -> None:
    st.markdown(
        """
        <style>
        .card {
          padding: 14px 16px;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.12);
          box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .card h3 { margin-bottom: 6px; }
        .muted { color: rgba(255,255,255,0.65); }
        </style>
        """,
        unsafe_allow_html=True,
    )


_card_css()


def _df_col_map(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}


def extract_sessions_per_day(df: pd.DataFrame) -> Optional[float]:
    cols = _df_col_map(df)
    if "sessions" not in cols:
        # Try a couple common variants.
        for alt in ["session_count", "session", "times_opened"]:
            if alt in cols:
                cols["sessions"] = cols[alt]
                break
        else:
            return None

    sessions_col = cols["sessions"]
    if "date" in cols:
        daily = df.groupby(cols["date"])[sessions_col].mean().dropna()
        if daily.empty:
            return None
        return float(daily.mean())

    # No date column; use global mean.
    return float(pd.to_numeric(df[sessions_col], errors="coerce").dropna().mean())


@st.cache_resource(show_spinner=False)
def get_model():
    model, metrics = train_risk_model(n_samples=4500, seed=42)
    return model, metrics


def plot_addiction_score_gauge(score: float) -> plt.Figure:
    score = float(np.clip(score, 0, 100))
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background segments.
    segments = [(0, 35, "#2ecc71"), (35, 65, "#f1c40f"), (65, 100, "#e74c3c")]
    for a, b, color in segments:
        ax.plot([a, b], [0.5, 0.5], linewidth=18, color=color, solid_capstyle="butt")

    # Needle.
    ax.plot([score, score], [0.5 - 0.35, 0.5 + 0.35], linewidth=3, color="white")
    ax.scatter([score], [0.5], s=50, color="white", zorder=5)
    ax.text(0, 0.1, "Low", color="#2ecc71", fontsize=10)
    ax.text(50, 0.1, "Moderate", color="#f1c40f", fontsize=10, ha="center")
    ax.text(100, 0.1, "High", color="#e74c3c", fontsize=10, ha="right")
    ax.text(score, 0.85, f"{score:.0f}", color="white", fontsize=13, ha="center")
    return fig


def plot_probability_bars(proba: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    levels = RISK_LABELS
    ax.bar(levels, proba, color=["#2ecc71", "#f1c40f", "#e74c3c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability")
    ax.set_title("ML Risk Prediction (3-class)")
    for i, v in enumerate(proba):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color="black", fontsize=10)
    fig.tight_layout()
    return fig


def plot_daily_line(daily: pd.Series, max_points: int = 40) -> plt.Figure:
    daily = daily.dropna()
    if daily.empty:
        return plt.figure()
    if len(daily) > max_points:
        daily = daily.iloc[-max_points:]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(daily.index, daily.values, marker="o", linewidth=2)
    ax.set_title("Daily Social Media Screen Time")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_platform_bar(platform_breakdown: pd.Series) -> plt.Figure:
    if platform_breakdown is None or platform_breakdown.empty:
        return plt.figure()
    top = platform_breakdown.head(10).iloc[::-1]  # reverse for horizontal bar ordering
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(top.index.astype(str), top.values, color="#7aa7ff")
    ax.set_xlabel("Total hours")
    ax.set_title("Top Platforms by Screen Time")
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Social Media Addiction Analyzer")
    st.caption("Interactive dashboard for screen-time analysis, addiction scoring, and ML risk prediction.")

    st.sidebar.header("Input Options")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your screen-time CSV (optional)",
        type=["csv"],
        accept_multiple_files=False,
    )
    use_csv = uploaded_file is not None

    daily_series: Optional[pd.Series] = None
    platform_breakdown: Optional[pd.Series] = None
    csv_avg_hours: Optional[float] = None
    csv_sessions_guess: Optional[float] = None

    df_preview = None
    if use_csv:
        try:
            raw = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(raw))
            df_preview = df.head(10)

            daily_series = try_extract_time_series(df)
            platform_breakdown = extract_platform_breakdown(df)
            if daily_series is not None and not daily_series.empty:
                csv_avg_hours = float(daily_series.mean())
            csv_sessions_guess = extract_sessions_per_day(df)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            use_csv = False

    # Manual inputs (always available; used by default for ML).
    st.sidebar.subheader("Quick Self-Assessment (Manual)")
    prefill_avg = csv_avg_hours if csv_avg_hours is not None else 3.0
    prefill_sessions = csv_sessions_guess if csv_sessions_guess is not None else 6.0
    prefill_before_bed = 1.0

    avg_social_hours_per_day = st.sidebar.slider(
        "Avg social media hours per day",
        min_value=0.0,
        max_value=12.0,
        value=float(np.clip(prefill_avg, 0.0, 12.0)),
        step=0.25,
    )
    sessions_per_day = st.sidebar.slider(
        "Estimated sessions per day",
        min_value=0.0,
        max_value=30.0,
        value=float(np.clip(prefill_sessions, 0.0, 30.0)),
        step=0.5,
    )
    before_bed_hours = st.sidebar.slider(
        "Hours before bed using social media",
        min_value=0.0,
        max_value=6.0,
        value=prefill_before_bed,
        step=0.25,
    )

    st.sidebar.divider()
    cannot_cut_down = st.sidebar.checkbox("Hard to cut down", value=False)
    affects_sleep = st.sidebar.checkbox("Affects sleep", value=False)
    impacts_work_study = st.sidebar.checkbox("Impacts work/study", value=False)
    craving_irritability = st.sidebar.checkbox("Irritable/Craving when trying to stop", value=False)

    feat = Features(
        avg_social_hours_per_day=avg_social_hours_per_day,
        sessions_per_day=sessions_per_day,
        before_bed_hours=before_bed_hours,
        cannot_cut_down=int(cannot_cut_down),
        affects_sleep=int(affects_sleep),
        impacts_work_study=int(impacts_work_study),
        craving_irritability=int(craving_irritability),
    )
    score = score_addiction(feat)
    score_class = score_to_class(score)

    st.markdown("---")
    tabs = st.tabs(["Dashboard", "Screen Time Analysis", "Charts", "ML Prediction", "How it Works"])

    with tabs[0]:
        left, right = st.columns([1, 2])
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Addiction Score")
            st.metric(label="Score (0-100)", value=f"{score:.0f}")
            st.markdown(f'<div class="muted">Risk level: <b>{RISK_LABELS[score_class]}</b></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            fig = plot_addiction_score_gauge(score)
            st.pyplot(fig, use_container_width=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("What drives your score")
            st.write(
                {
                    "Avg hours/day": round(avg_social_hours_per_day, 2),
                    "Sessions/day": round(sessions_per_day, 2),
                    "Before-bed hours": round(before_bed_hours, 2),
                    "Hard to cut down": cannot_cut_down,
                    "Affects sleep": affects_sleep,
                    "Impacts work/study": impacts_work_study,
                    "Irritable/Craving": craving_irritability,
                }
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Screen time analysis")
        if not use_csv:
            st.info("Upload a CSV for richer analysis; the app will still work with manual inputs.")
        else:
            st.write("CSV preview (first rows):")
            st.dataframe(df_preview)

            if daily_series is None:
                st.warning(
                    "Could not find a `date` column and a compatible hours column in your CSV. "
                    "Supported examples: `date, platform, hours` or `date, social_hours`."
                )
            else:
                st.write("Daily summary (from CSV):")
                daily = daily_series.dropna()
                summary = {
                    "Days analyzed": int(len(daily)),
                    "Avg hours/day": float(daily.mean()),
                    "Peak hours/day": float(daily.max()),
                    "Min hours/day": float(daily.min()),
                }
                st.write(summary)

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Charts (matplotlib)")

        if platform_breakdown is not None and not platform_breakdown.empty:
            st.pyplot(plot_platform_bar(platform_breakdown), use_container_width=True)
        else:
            st.info("Platform breakdown chart needs `platform` and `hours` (or `platform` and `social_hours`) in your CSV.")

        if daily_series is not None and not daily_series.empty:
            st.pyplot(plot_daily_line(daily_series), use_container_width=True)
        else:
            st.info("Daily line chart needs a `date` column plus daily social hours in your CSV.")

        # Simple distribution using the manual score inputs.
        st.markdown("### Quick distribution (based on your inputs)")
        bins = np.linspace(0, 100, 11)
        # Spread one sample into neighboring bins (visual only).
        counts = np.zeros_like(bins[:-1], dtype=float)
        idx = np.clip(np.searchsorted(bins, score) - 1, 0, len(counts) - 1)
        counts[idx] = 1.0
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar([f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(counts))], counts, color="#7aa7ff")
        ax.set_ylabel("Presence")
        ax.set_xlabel("Addiction score bin")
        ax.set_title("Your score falls into one bin")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ML prediction (scikit-learn)")

        model, metrics = get_model()
        st.caption(
            "Model is trained on synthetic data generated from the heuristic scoring rules. "
            "If you have real labeled data, this can be adapted."
        )
        st.write(f"Validation accuracy on synthetic holdout: **{metrics['accuracy']:.3f}**")
        with st.expander("Show classification report (synthetic)"):
            st.text(metrics["classification_report"])

        proba = predict_risk_proba(model, feat)
        pred_class = class_from_proba(proba)

        st.markdown(
            f"Predicted risk: **{RISK_LABELS[pred_class]}**"
        )
        st.pyplot(plot_probability_bars(proba), use_container_width=True)

        st.markdown('<div class="muted">Tip: try toggling the checkboxes and moving sliders to see how probabilities change.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How it works (high level)")
        st.markdown(
            """
            - **Screen time analysis:** If your CSV contains `date` and hours (e.g. `platform + hours` or `social_hours`), the app aggregates total hours per day and per platform.
            - **Addiction score:** A transparent weighted heuristic combines time spent (hours, sessions, before-bed hours) with self-assessment indicators (cut-down difficulty, sleep impact, work/study impact, craving/irritability).
            - **ML prediction:** A `RandomForestClassifier` is trained on **synthetic** examples generated by sampling realistic ranges for the inputs, then labeling them using the same heuristic score into `Low / Moderate / High`.
            
            This tool is for awareness only and is not a medical diagnosis.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

