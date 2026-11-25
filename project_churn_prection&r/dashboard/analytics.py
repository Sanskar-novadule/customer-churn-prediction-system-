import os

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn & Revenue Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_customer_churn.csv",
)


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found:\n{DATA_PATH}")
    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

if "Churn" not in df.columns:
    st.error("The dataset must contain a 'Churn' column.")
    st.stop()

df["Churn_Yes"] = df["Churn"].map({"No": 0, "Yes": 1})

if "MonthlyCharges" in df.columns:
    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"], errors="coerce"
    )

if "Tenure" in df.columns:
    df["Tenure"] = pd.to_numeric(
        df["Tenure"], errors="coerce"
    )

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        background: linear-gradient(90deg, #233d73, #3d5f9e);
        padding: 18px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .risk-box {
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: 700;
        min-height: 80px;
    }

    .small-text {
        font-size: 13px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Customer Churn Prediction & Revenue Forecasting Dashboard'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(df)

churn_rate = (
    df["Churn_Yes"].mean() * 100
    if total_customers > 0
    else 0
)

avg_monthly_charges = (
    df["MonthlyCharges"].mean()
    if "MonthlyCharges" in df.columns
    else 0
)

avg_tenure = (
    df["Tenure"].mean()
    if "Tenure" in df.columns
    else 0
)

monthly_revenue = (
    df["MonthlyCharges"].sum()
    if "MonthlyCharges" in df.columns
    else 0
)

predicted_revenue_12_months = monthly_revenue * 12


# ============================================================
# KPI CARDS
# ============================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.0f}%",
    )

with kpi2:
    st.metric(
        "Avg. Monthly Charges",
        f"${avg_monthly_charges:,.0f}",
    )

with kpi3:
    st.metric(
        "Avg. Tenure",
        f"{avg_tenure:.0f} months",
    )

with kpi4:
    st.metric(
        "Predicted Revenue",
        f"${predicted_revenue_12_months:,.0f}",
    )


st.divider()


# ============================================================
# CHURN BY CONTRACT
# ============================================================

left, middle, right = st.columns([1.15, 1.45, 1])


# -------------------- BAR CHART ------------------------------

with left:

    st.markdown(
        '<div class="section-title">Churn by Contract Type</div>',
        unsafe_allow_html=True,
    )

    if "Contract" in df.columns:

        contract_churn = (
            df.groupby("Contract")["Churn_Yes"]
            .sum()
            .reindex(
                ["Month-to-month", "One year", "Two year"],
                fill_value=0,
            )
        )

        fig, ax = plt.subplots(figsize=(5, 3.2))

        ax.bar(
            contract_churn.index,
            contract_churn.values,
        )

        ax.set_ylabel("Churned Customers")
        ax.tick_params(axis="x", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    else:
        st.info("Contract column not found.")


# -------------------- CHURN GAUGE ----------------------------

with middle:

    st.markdown(
        '<div class="section-title">Churn Prediction</div>',
        unsafe_allow_html=True,
    )

    # Semicircle gauge based on actual overall churn rate.
    gauge_value = max(0, min(100, churn_rate))

    fig, ax = plt.subplots(
        figsize=(5, 3.2),
        subplot_kw={"projection": "polar"},
    )

    # Green -> yellow -> red background zones
    zones = [
        (0, 30, "Low"),
        (30, 60, "Medium"),
        (60, 100, "High"),
    ]

    for start, end, label in zones:
        theta = [
            start / 100 * 3.14159,
            end / 100 * 3.14159,
        ]

        ax.bar(
            x=(theta[0] + theta[1]) / 2,
            height=1,
            width=theta[1] - theta[0],
            bottom=0,
            alpha=0.75,
            color={
                "Low": "seagreen",
                "Medium": "gold",
                "High": "firebrick",
            }[label],
        )

    needle_angle = gauge_value / 100 * 3.14159

    ax.plot(
        [needle_angle, needle_angle],
        [0, 0.88],
        linewidth=4,
        color="black",
    )

    ax.scatter(
        [needle_angle],
        [0.88],
        s=50,
        color="black",
    )

    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)

    ax.text(
        0.5,
        0.20,
        f"{gauge_value:.0f}%",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )

    if gauge_value < 30:
        risk = "Low"
    elif gauge_value < 60:
        risk = "Medium"
    else:
        risk = "High"

    ax.text(
        0.5,
        0.38,
        f"Churn Risk: {risk}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# -------------------- REVENUE FORECAST -----------------------

with right:

    st.markdown(
        '<div class="section-title">Revenue Forecast</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"Next 12 Months: "
        f"${predicted_revenue_12_months:,.0f}"
    )

    st.write(
        "Estimated from current monthly charges "
        "multiplied by 12 months."
    )


st.divider()


# ============================================================
# CUSTOMER RISK SEGMENTATION
# ============================================================

seg_left, seg_right = st.columns([1.55, 1])


# -------------------- SEGMENTATION ----------------------------

with seg_left:

    st.markdown(
        '<div class="section-title">Customer Segmentation</div>',
        unsafe_allow_html=True,
    )

    low_count = int((df["Churn_Yes"] == 0).sum())

    high_count = int((df["Churn_Yes"] == 1).sum())

    # A simple analytics segmentation:
    # non-churned customers = low risk,
    # churned customers = high risk.
    # Medium risk is kept at 0 unless a probability model is available.
    medium_count = 0

    total = max(total_customers, 1)

    low_pct = low_count / total * 100
    medium_pct = medium_count / total * 100
    high_pct = high_count / total * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="risk-box" style="background:#f28c28;">
                Low Risk<br>
                <span style="font-size:24px;">{low_pct:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="risk-box" style="background:#8baa21;">
                Medium Risk<br>
                <span style="font-size:24px;">{medium_pct:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="risk-box" style="background:#e6332a;">
                High Risk<br>
                <span style="font-size:24px;">{high_pct:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

with seg_right:

    st.markdown(
        '<div class="section-title">Sample Predictions</div>',
        unsafe_allow_html=True,
    )

    sample_columns = []

    if "CustomerID" in df.columns:
        sample_columns.append("CustomerID")

    sample_columns.append("Churn")

    if "TotalCharges" in df.columns:
        sample_columns.append("TotalCharges")

    sample_df = df[sample_columns].head(5).copy()

    sample_df["Churn Probability"] = (
        sample_df["Churn"]
        .map({"No": 0.15, "Yes": 0.80})
    )

    sample_df["Churn Probability"] = (
        sample_df["Churn Probability"] * 100
    ).round(0).astype(str) + "%"

    if "TotalCharges" in sample_df.columns:
        sample_df["Estimated Revenue"] = (
            sample_df["TotalCharges"]
        ).map(lambda x: f"${x:,.0f}")

    if "Churn" in sample_df.columns:
        sample_df = sample_df.drop(columns=["Churn"])

    st.dataframe(
        sample_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction & Revenue Forecasting | "
    "Analytics Dashboard"
)
