"""
app.py

ScamLens AI — Streamlit front-end.
All prediction / rule-based logic lives in utils.py; this file is UI only.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import utils

# Page config

st.set_page_config(
    page_title="ScamLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached dataset stats for the Home page

@st.cache_data
def load_stats():
    df = pd.read_csv("scam_dataset.csv")
    return {
        "total": len(df),
        "avg_loss": df["amount_inr"].mean(),
        "top_fraud_type": df["fraud_type"].mode()[0],
        "top_upi_app": df["upi_app"].mode()[0],
    }

stats = load_stats()

@st.cache_data
def load_eda_df():
    return utils.load_featured_df()

@st.cache_data
def load_metrics():
    return utils.get_model_metrics()

# Consistent color palette for every chart (matches the CSS theme)
CHART_COLORS = ["#6C7BFF", "#16C7A6", "#F5A623", "#E4483A", "#3E8EF7", "#B983FF", "#F26D3D", "#2FD3C6", "#9AA6D1", "#FF7AB6", "#5AD1A0"]

def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#EAF0FF"),
        title_font=dict(family="Poppins, sans-serif", size=16),
        margin=dict(t=50, l=10, r=10, b=10),
        height=height,
    )
    return fig

# Global CSS — design tokens
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
    --bg-deep:#0A1530;
    --bg-deep2:#122048;
    --card-glass: rgba(255,255,255,0.055);
    --card-glass-border: rgba(255,255,255,0.12);
    --card-light:#FFFFFF;
    --ink:#EAF0FF;
    --ink-dim:#9FB0D9;
    --teal:#16C7A6;
    --amber:#F5A623;
    --red:#E4483A;
    --violet:#6C7BFF;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    color: var(--ink);
}

/* App background: deep navy gradient + faint circuit/network pattern*/
.stApp{
    background:
        radial-gradient(circle at 15% 10%, rgba(108,123,255,0.18), transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(22,199,166,0.14), transparent 45%),
        linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-deep2) 55%, var(--bg-deep) 100%);
    background-attachment: fixed;
}
.stApp::before{
    content:"";
    position:fixed; inset:0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><g fill='none' stroke='%234A5C9B' stroke-width='0.6' opacity='0.18'><circle cx='60' cy='60' r='2.4'/><path d='M60 0 V120 M0 60 H120 M15 15 L105 105 M105 15 L15 105'/></g></svg>");
    background-size: 120px 120px;
    filter: blur(1px);
    pointer-events:none;
    z-index:0;
}

h1,h2,h3{ font-family:'Poppins', sans-serif; letter-spacing:-0.01em; }

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0C1836 0%, #0A1530 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/*  Glass card  */
.glass-card{
    background: var(--card-glass);
    border: 1px solid var(--card-glass-border);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* Light card (used on report, easier to read dense text)*/
.light-card{
    background: var(--card-light);
    color:#1A2138;
    border-radius:18px;
    padding:22px 24px;
    margin-bottom:18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border-left: 6px solid var(--violet);
}
.light-card h4{ margin-top:0; color:#1A2138; font-family:'Poppins',sans-serif;}
.light-card .mono{ font-family:'JetBrains Mono', monospace; }

/* Stat pill cards (Home) */
.stat-card{
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
    border:1px solid rgba(255,255,255,0.12);
    border-radius:16px;
    padding:18px 20px;
    text-align:center;
}
.stat-card .num{
    font-family:'JetBrains Mono', monospace;
    font-size:1.9rem; font-weight:700; color: var(--teal);
}
.stat-card .lbl{ color:var(--ink-dim); font-size:0.85rem; margin-top:4px;}

/* Badge */
.badge{
    display:inline-block; padding:6px 16px; border-radius:999px;
    font-weight:700; font-size:0.95rem; color:#fff; font-family:'Poppins',sans-serif;
}

/* Emergency buttons (Home) */
.emg-btn{
    display:block; text-decoration:none !important; border-radius:16px;
    padding:20px 22px; font-family:'Poppins',sans-serif; font-weight:700;
    font-size:1.05rem; color:#fff !important; text-align:center;
    box-shadow:0 10px 24px rgba(0,0,0,0.3); transition: transform 0.15s ease;
}
.emg-btn:hover{ transform: translateY(-3px); }
.emg-call{ background: linear-gradient(135deg, var(--red), #B8281D); }
.emg-report{ background: linear-gradient(135deg, var(--violet), #3E4FD8); }
.emg-btn .sub{ display:block; font-family:'Inter',sans-serif; font-weight:400; font-size:0.8rem; opacity:0.85; margin-top:4px;}

/* Risk meter (signature element) */
.risk-meter-track{
    position:relative; height:14px; border-radius:999px; margin:14px 0 30px 0;
    background: linear-gradient(90deg, #22C55E 0%, #F5A623 35%, #F26D3D 65%, #E4483A 100%);
}
.risk-meter-pointer{
    position:absolute; top:-10px; width:2px; height:34px; background:#1A2138;
}
.risk-meter-pointer::after{
    content: attr(data-label);
    position:absolute; top:-26px; left:50%; transform:translateX(-50%);
    font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700;
    background:#1A2138; color:#fff; padding:2px 8px; border-radius:6px; white-space:nowrap;
}

/* Resource link buttons */
.resource-link{
    display:flex; align-items:center; gap:10px; text-decoration:none !important;
    background:#F4F6FB; color:#1A2138 !important; border-radius:12px; padding:12px 16px;
    margin-bottom:10px; font-weight:600; border:1px solid #E3E7F3; transition: all 0.15s ease;
}
.resource-link:hover{ background:#E9ECFB; transform:translateX(3px); }

hr{ border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation

PAGES = [
    "🏠 Home",
    "ℹ️ About",
    "📊 EDA Dashboard",
    "🔍 Analyze Fraud",
    "📈 Model Insights",
    "📌 Insights & Conclusion"
]

if "page_index" not in st.session_state:
    st.session_state.page_index = 0


with st.sidebar:
    st.markdown("## 🛡️ ScamLens AI")
    st.caption("AI Recovery Assistant for UPI Fraud")

    page = st.radio(
        "Navigate",
        PAGES,
        index=st.session_state.page_index,
        label_visibility="collapsed"
    )

    st.session_state.page_index = PAGES.index(page)



# HOME PAGE

if page == "🏠 Home":
    st.markdown("<h1 style='font-size:2.6rem;margin-bottom:0;'>ScamLens AI 🛡️</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9FB0D9;font-size:1.05rem;max-width:640px;'>"
        "Lost money to a UPI scam? Get an instant AI-based severity check, "
        "understand the real financial impact on you, and know exactly who to contact next."
        "</p>", unsafe_allow_html=True
    )

    st.markdown("###  ")
    st.markdown("#### 🚨 In an emergency, act now")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<a class="emg-btn emg-call" href="tel:1930">📞 Call 1930'
            '<span class="sub">National Cyber Crime Helpline — 24x7</span></a>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<a class="emg-btn emg-report" href="https://cybercrime.gov.in" target="_blank">📝 Report Cyber Crime'
            '<span class="sub">File a complaint at cybercrime.gov.in</span></a>',
            unsafe_allow_html=True,
        )

    st.markdown("###  ")
    st.markdown("#### Fraud Landscape (from analyzed data)")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, lbl in [
        (s1, f"{stats['total']:,}", "Fraud Cases Analyzed"),
        (s2, f"₹{stats['avg_loss']:,.0f}", "Average Loss Amount"),
        (s3, stats["top_fraud_type"], "Most Common Fraud Type"),
        (s4, stats["top_upi_app"], "Most Targeted UPI App"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("###  ")
    st.markdown("#### How it works")
    h1, h2, h3 = st.columns(3)
    steps = [
        ("1️⃣", "Enter Transaction Details", "Amount lost, monthly income, time of fraud, UPI app, and how the scam happened."),
        ("2️⃣", "Get AI Severity + Impact", "A Decision Tree predicts severity; your income tells us the real damage."),
        ("3️⃣", "Get Recovery Guidance", "Specific next steps, government helplines, and a downloadable report."),
    ]
    for col, (icon, title, desc) in zip([h1, h2, h3], steps):
        with col:
            st.markdown(
                f'<div class="glass-card"><div style="font-size:1.6rem;">{icon}</div>'
                f'<h4 style="margin:8px 0 6px 0;">{title}</h4>'
                f'<div style="color:#9FB0D9;font-size:0.9rem;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("###  ")
    if st.button("🔍 Analyze My Fraud Case →", type="primary"):
        st.session_state.page_index = 3
        st.rerun()

# ANALYZE FRAUD PAGE



# ABOUT PAGE

elif page == "ℹ️ About":
    about_df = pd.read_csv("scam_dataset.csv")
 
    st.markdown("<h1 style='font-size:2.2rem;'>ℹ️ About ScamLens AI</h1>", unsafe_allow_html=True)
 
    # Project Overview
    st.markdown(
        '<div class="glass-card">'
        '<h4 style="margin-top:0;">📖 Project Overview</h4>'
        '<p>ScamLens AI is a student project that analyzes UPI fraud transaction patterns '
        'and helps victims understand two things a raw rupee amount cannot tell them on its own: '
        '<b>how severe this fraud is</b>, and <b>how much it really impacts them</b> relative to their income.</p>'
        '<p>A Decision Tree classifier predicts <b>Fraud Severity</b> from the transaction itself. Separately '
        '— because it is a direct calculation, not something that needs to be learned — the app computes a '
        '<b>Loss-to-Income Ratio</b> and maps it to a <b>Financial Impact</b> level, then routes the user '
        'to a specific recommended action and official government resources.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
 
    #  Objective + Problem Statement
    o1, o2 = st.columns(2)
    with o1:
        st.markdown(
            '<div class="light-card" style="border-left-color:var(--teal);"><h4>🎯 Objective</h4>'
            '<ul style="padding-left:18px;margin:0;">'
            '<li>Analyze real-world patterns behind UPI fraud transactions.</li>'
            '<li>Engineer meaningful features (time slot, day type, amount category).</li>'
            '<li>Build an explainable ML model to classify fraud severity.</li>'
            '<li>Go beyond classification — quantify real financial impact and guide recovery.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            '<div class="light-card" style="border-left-color:var(--red);"><h4>❓ Problem Statement</h4>'
            '<p style="margin:0;">Existing fraud-awareness tools tell a victim a rupee amount was lost — '
            'nothing more. They ignore that the <i>same</i> loss can be trivial for one person and financially '
            'devastating for another, and they offer no next step. ScamLens AI addresses both gaps: it '
            'classifies severity from the transaction pattern, adjusts impact for the victim\'s income, and '
            'points them to concrete recovery action.</p></div>',
            unsafe_allow_html=True,
        )
 
    # Dataset Information 
    st.markdown("###  ")
    st.markdown("#### 📂 Dataset Information")
    st.markdown(
        '<div class="light-card">'
        '<table style="width:100%;font-size:0.92rem;border-collapse:collapse;">'
        '<tr><td style="padding:5px 0;width:220px;"><b>Dataset Name</b></td><td>scam_dataset.csv — UPI Fraud Transactions</td></tr>'
        f'<tr><td style="padding:5px 0;"><b>Number of Records</b></td><td>{len(about_df):,} fraud transactions</td></tr>'
        f'<tr><td style="padding:5px 0;"><b>Number of Features</b></td><td>{about_df.shape[1]} raw columns (10 engineered columns added downstream)</td></tr>'
        '<tr><td style="padding:5px 0;"><b>Data Source</b></td><td>Simulated / curated dataset representing realistic Indian UPI fraud patterns (academic project use)</td></tr>'
        f'<tr><td style="padding:5px 0;"><b>Date Range</b></td><td>{about_df["date"].min()[:4]}–{about_df["date"].max()[:4]}</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )
 
    a1, a2, a3, a4 = st.columns(4)
    for col, num, lbl in [
        (a1, f"{len(about_df):,}", "Total Records"),
        (a2, f"{about_df['fraud_type'].nunique()}", "Fraud Types"),
        (a3, f"{about_df['upi_app'].nunique()}", "UPI Apps Covered"),
        (a4, f"{about_df['fraud_lure'].nunique()}", "Distinct Fraud Lures"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
 
    st.markdown("###  ")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            '<div class="light-card"><h4>📑 Key Columns</h4>'
            '<table style="width:100%;font-size:0.88rem;border-collapse:collapse;">'
            '<tr style="border-bottom:1px solid #E3E7F3;"><th style="text-align:left;padding:6px 0;">Column</th><th style="text-align:left;">Meaning</th></tr>'
            '<tr><td style="padding:5px 0;"><code>transaction_id</code></td><td>Unique transaction reference</td></tr>'
            '<tr><td style="padding:5px 0;"><code>date / time</code></td><td>When the fraud occurred</td></tr>'
            '<tr><td style="padding:5px 0;"><code>fraud_type</code></td><td>Category of scam (Phishing, Vishing, etc.)</td></tr>'
            '<tr><td style="padding:5px 0;"><code>fraud_lure</code></td><td>The pretext used to trick the victim</td></tr>'
            '<tr><td style="padding:5px 0;"><code>amount_inr</code></td><td>Amount lost, in ₹</td></tr>'
            '<tr><td style="padding:5px 0;"><code>upi_app</code></td><td>UPI app used for the transaction</td></tr>'
            '<tr><td style="padding:5px 0;"><code>transaction_type</code></td><td>P2P, P2M, or Collect Request</td></tr>'
            '<tr><td style="padding:5px 0;"><code>sender_upi_id</code></td><td>Anonymized sender UPI handle</td></tr>'
            '</table></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="light-card"><h4>🔎 Raw Data Preview</h4>', unsafe_allow_html=True)
        st.dataframe(about_df.head(8), use_container_width=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)
 
    # Machine Learning Model
    st.markdown("###  ")
    st.markdown("#### 🤖 Machine Learning Model")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            '<div class="light-card" style="border-left-color:var(--violet);"><h4>Model Details</h4>'
            '<table style="width:100%;font-size:0.92rem;">'
            '<tr><td style="padding:4px 0;width:160px;"><b>Algorithm</b></td><td>Decision Tree Classifier</td></tr>'
            '<tr><td style="padding:4px 0;"><b>Criterion</b></td><td>Entropy (Information Gain)</td></tr>'
            '<tr><td style="padding:4px 0;"><b>Max Depth</b></td><td>5</td></tr>'
            '<tr><td style="padding:4px 0;"><b>Target Variable</b></td><td>Fraud_Severity (Low / Medium / High)</td></tr>'
            '</table></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="light-card" style="border-left-color:var(--amber);"><h4>Input Features</h4>'
            '<ul style="padding-left:18px;margin:0;font-size:0.92rem;">'
            '<li><code>amount_inr</code> — transaction amount</li>'
            '<li><code>Time_Slot</code> — Morning / Afternoon / Evening / Night</li>'
            '<li><code>Day_Type</code> — Weekday / Weekend</li>'
            '<li><code>upi_app</code> — UPI app used</li>'
            '<li><code>transaction_type</code> — P2P / P2M / Collect Request</li>'
            '<li><code>fraud_lure</code> — pretext used to trick the victim</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
 
    #  Workflow (arrow-connected flow)
    st.markdown("###  ")
    st.markdown("#### ⚙️ Project Workflow")
    workflow_steps = ["Dataset", "Data Cleaning", "Feature Engineering", "EDA", "Decision Tree Model", "Fraud Analysis"]
    flow_html = '<div class="glass-card"><div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;justify-content:center;">'
    for i, step in enumerate(workflow_steps):
        flow_html += (
            f'<div style="background:linear-gradient(135deg, var(--violet), #3E4FD8);color:#fff;'
            f'padding:12px 18px;border-radius:12px;font-weight:600;font-family:Poppins,sans-serif;'
            f'font-size:0.9rem;white-space:nowrap;box-shadow:0 6px 16px rgba(0,0,0,0.25);">{step}</div>'
        )
        if i < len(workflow_steps) - 1:
            flow_html += '<div style="color:var(--teal);font-size:1.4rem;">→</div>'
    flow_html += "</div></div>"
    st.markdown(flow_html, unsafe_allow_html=True)
 
    st.markdown("###  ")
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1", "Data Cleaning", "Parsed dates/times, standardized amount formatting."),
        ("2", "Feature Engineering", "Derived Time_Slot, Day_Type, Amount_Category, Fraud_Severity."),
        ("3", "ML Model", "Decision Tree (entropy, depth 5) predicts Fraud_Severity."),
        ("4", "Recovery Layer", "Income-adjusted impact, recommended action, government resources."),
    ]
    for col, (num, title, desc) in zip([p1, p2, p3, p4], steps):
        with col:
            st.markdown(
                f'<div class="glass-card"><div style="color:var(--teal);font-family:JetBrains Mono, monospace;font-weight:700;">STEP {num}</div>'
                f'<h4 style="margin:6px 0;">{title}</h4>'
                f'<div style="color:#9FB0D9;font-size:0.85rem;">{desc}</div></div>',
                unsafe_allow_html=True,
            )
 
    # Technologies Used
    st.markdown("###  ")
    st.markdown("#### 🛠 Technologies Used")
    tech_list = [
        ("Python", "Core language"), ("Pandas", "Data wrangling"), ("NumPy", "Numerical computing"),
        ("Scikit-learn", "Decision Tree model"), ("Plotly", "Interactive charts"),
        ("Streamlit", "Web app framework"), ("Joblib", "Model serialization"), ("fpdf2", "PDF report generation"),
    ]
    tech_html = '<div class="glass-card"><div style="display:flex;flex-wrap:wrap;gap:12px;">'
    for name, desc in tech_list:
        tech_html += (
            f'<div style="background:rgba(108,123,255,0.12);border:1px solid var(--violet);border-radius:10px;'
            f'padding:8px 14px;min-width:130px;">'
            f'<div style="font-weight:700;color:#C7CCFF;font-family:Poppins,sans-serif;font-size:0.9rem;">{name}</div>'
            f'<div style="color:#9FB0D9;font-size:0.75rem;">{desc}</div></div>'
        )
    tech_html += "</div></div>"
    st.markdown(tech_html, unsafe_allow_html=True)
 
    st.markdown(
        '<p style="color:#9FB0D9;font-size:0.85rem;margin-top:18px;">'
        'Note: Monthly income data is not present in the original dataset (for privacy reasons) and was '
        'simulated for demonstration during model development. In this app, you enter your own income directly.'
        '</p>',
        unsafe_allow_html=True,
    )




# EDA DASHBOARD PAGE

elif page == "📊 EDA Dashboard":
    eda_df = load_eda_df()

    st.markdown("<h1 style='font-size:2.2rem;'>📊 EDA Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB0D9;'>Interactive exploration of 1,000 UPI fraud cases.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎣 Fraud Types & Lures", "📲 Apps & Transactions", "🕒 Time Patterns",
        "🔗 Relationships", "💰 Financial Impact",
    ])

    # TAB 1: Fraud Types & Lures
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            ft = eda_df["fraud_type"].value_counts().reset_index()
            ft.columns = ["Fraud Type", "Count"]
            fig = px.bar(ft, x="Fraud Type", y="Count", color="Fraud Type", text="Count",
                         title="Most Common Fraud Types", color_discrete_sequence=CHART_COLORS)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fl = eda_df["fraud_lure"].value_counts().reset_index()
            fl.columns = ["Fraud Lure", "Count"]
            fig = px.bar(fl, x="Count", y="Fraud Lure", orientation="h", color="Fraud Lure",text="Count",
                          title="Fraud Lure Distribution", color_discrete_sequence=CHART_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), width="stretch")

        lure_amt = eda_df.groupby("fraud_lure")["amount_inr"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(lure_amt, x="fraud_lure", y="amount_inr", color="fraud_lure", text="amount_inr",
                     title="Top Fraud Lures by Total Amount Lost (₹)", color_discrete_sequence=CHART_COLORS)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    # TAB 2: Apps & Transactions
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            app = eda_df["upi_app"].value_counts().reset_index()
            app.columns = ["UPI App", "Count"]
            fig = px.pie(app, names="UPI App", values="Count", hole=0.55, title="Most Used UPI App",
                         color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            tt = eda_df["transaction_type"].value_counts().reset_index()
            tt.columns = ["Transaction Type", "Count"]
            fig = px.pie(tt, names="Transaction Type", values="Count", hole=0.55,
                         title="Transaction Type Distribution", color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            upi_avg = eda_df.groupby("upi_app")["amount_inr"].mean().sort_values().reset_index()
            fig = px.bar(upi_avg, x="amount_inr", y="upi_app", orientation="h", color="upi_app",
                         title="Average Fraud Amount by UPI App", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c4:
            fraud_avg = eda_df.groupby("fraud_type")["amount_inr"].mean().sort_values().reset_index()
            fig = px.bar(fraud_avg, x="amount_inr", y="fraud_type", orientation="h", color="fraud_type",
                         title="Average Fraud Amount by Fraud Type", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # TAB 3: Time Patterns
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            day = eda_df["Day_Name"].value_counts().reset_index()
            day.columns = ["Day", "Count"]
            fig = px.bar(day, x="Day", y="Count", color="Day", text="Count",
                         title="Day-wise Fraud Count", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            dt = eda_df["Day_Type"].value_counts().reset_index()
            dt.columns = ["Day Type", "Count"]
            fig = px.pie(dt, names="Day Type", values="Count", hole=0.55, title="Weekend vs Weekday",
                         color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            amt = eda_df["Amount_Category"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
            amt.columns = ["Amount Category", "Count"]
            fig = px.bar(amt, x="Amount Category", y="Count", color="Amount Category", text="Count",
                         title="Amount Category Distribution",
                         color_discrete_sequence=["#22C55E", "#F5A623", "#E4483A"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c4:
            ts = eda_df["Time_Slot"].value_counts().reindex(["Morning", "Afternoon", "Evening", "Night"]).reset_index()
            ts.columns = ["Time Slot", "Count"]
            fig = px.bar(ts, x="Time Slot", y="Count", color="Time Slot", text="Count",
                         title="Time Slot Distribution", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # TAB 4: Relationships
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            ct = pd.crosstab(eda_df["fraud_type"], eda_df["transaction_type"])
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Sunset",
                             title="Fraud Type vs Transaction Type")
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
        with c2:
            ct2 = pd.crosstab(eda_df["fraud_type"], eda_df["upi_app"])
            fig = px.imshow(ct2, text_auto=True, color_continuous_scale="Purp",
                             title="Fraud Type vs UPI App")
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

        fig = px.scatter(eda_df, x="amount_inr", y="fraud_type", color="upi_app", size="amount_inr",
                          hover_name="fraud_lure", title="Bubble Chart of Fraud Transactions",
                          color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True)

        fig = px.histogram(eda_df, x="upi_app", color="Fraud_Severity", barmode="group",
                            title="Fraud Severity by UPI App",
                            color_discrete_map={"Low": "#22C55E", "Medium": "#F5A623", "High": "#E4483A"})
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("#### 🏆 Top 10 Largest Frauds")
        top10 = eda_df.sort_values("amount_inr", ascending=False).head(10)[
            ["transaction_id", "fraud_type", "fraud_lure", "amount_inr", "upi_app", "Fraud_Severity"]
        ]
        st.dataframe(top10, use_container_width=True)

    # TAB 5: Financial Impact (our project's core insight)
    with tab5:
        st.markdown(
            '<div class="glass-card" style="border-left:4px solid var(--teal);">'
            "This tab is why ScamLens AI doesn't stop at severity: the same rupee amount "
            "hits different people differently, so impact is measured relative to income."
            "</div>", unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            fi = eda_df["Financial_Impact"].value_counts().reindex(["Low", "Moderate", "High", "Critical"]).reset_index()
            fi.columns = ["Financial Impact", "Count"]
            fig = px.bar(fi, x="Financial Impact", y="Count", color="Financial Impact", text="Count",
                         title="Financial Impact Distribution (Income-Adjusted)",
                         color_discrete_map={"Low": "#22C55E", "Moderate": "#F5A623", "High": "#F26D3D", "Critical": "#E4483A"})
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            ct3 = pd.crosstab(eda_df["Fraud_Severity"], eda_df["Financial_Impact"])
            ct3 = ct3.reindex(index=["Low", "Medium", "High"], columns=["Low", "Moderate", "High", "Critical"])
            fig = px.imshow(ct3, text_auto=True, color_continuous_scale="Magma",
                             title="Amount-Based Severity vs Income-Based Impact")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption(
            "Notice the diagonal isn't perfectly clean — a 'Low' amount-severity case can still be a "
            "'Critical' financial impact for a low-income victim. That mismatch is the whole point."
        )

elif page == "🔍 Analyze Fraud":
    st.markdown("<h1 style='font-size:2.2rem;'>🔍 Analyze Fraud</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB0D9;'>Fill in what happened — this stays on your screen only.</p>", unsafe_allow_html=True)

    with st.form("analyze_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("💰 Amount Lost (₹)", min_value=1.0, value=5000.0, step=100.0)
            monthly_income = st.number_input("💵 Your Monthly Income (₹)", min_value=1.0, value=25000.0, step=500.0)
            upi_app = st.selectbox("📲 UPI App Used", utils.UPI_APP_OPTIONS)
            transaction_type = st.selectbox("💳 Transaction Type", utils.TRANSACTION_TYPE_OPTIONS)
        with col2:
            time_slot = st.selectbox("🕒 Time of Day", utils.TIME_SLOT_OPTIONS)
            day_type = st.selectbox("📅 Day Type", utils.DAY_TYPE_OPTIONS)
            fraud_lure = st.selectbox("🎣 How Were You Lured?", utils.FRAUD_LURE_OPTIONS)

        submitted = st.form_submit_button("🤖 Generate AI Report", type="primary", use_container_width=True)

    if submitted:
        st.session_state["result"] = utils.analyze_fraud(
            amount, monthly_income, time_slot, day_type, upi_app, transaction_type, fraud_lure
        )

    # AI REPORT
    if "result" in st.session_state:
        r = st.session_state["result"]
        sev_color = utils.SEVERITY_COLORS[r["severity"]]
        imp_color = utils.IMPACT_COLORS[r["impact"]]

        st.markdown("---")
        st.markdown("## 📋 AI Report")

        # Risk Summary
        st.markdown(
            f'<div class="light-card" style="border-left-color:{sev_color};">'
            f'<h4>🎯 Risk Summary</h4>'
            f'<span class="badge" style="background:{sev_color};">Fraud Severity: {r["severity"]}</span> '
            f'<span class="badge" style="background:{imp_color};margin-left:8px;">Financial Impact: {r["impact"]}</span>'
            f'<p style="margin-top:12px;">Likely fraud type: <b>{r["fraud_type"]}</b> via <b>{r["fraud_lure"]}</b></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Risk meter (signature visual — loss as % of income)
        pointer_pos = min(max(r["loss_ratio"], 0), 100)
        st.markdown(
            f'<div class="light-card"><h4>📊 Loss vs. Your Income</h4>'
            f'<div class="mono" style="font-size:0.85rem;color:#555;">This loss equals <b>{r["loss_ratio"]}%</b> of your monthly income</div>'
            f'<div class="risk-meter-track">'
            f'<div class="risk-meter-pointer" data-label="{r["loss_ratio"]}%" style="left:{pointer_pos}%;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888;">'
            f'<span>0% (Low)</span><span>15%</span><span>40%</span><span>100%+ (Critical)</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Recommended Action
        st.markdown(
            f'<div class="light-card"><h4>✅ Recommended Action</h4>'
            f'<p>{r["recommended_action"]}</p></div>',
            unsafe_allow_html=True,
        )

        # Emergency Steps (only if High/Critical impact)
        if r["is_emergency"]:
            steps_html = "".join(f"<li style='margin-bottom:6px;'>{s}</li>" for s in utils.EMERGENCY_STEPS)
            st.markdown(
                f'<div class="light-card" style="border-left-color:{utils.IMPACT_COLORS["Critical"]};">'
                f'<h4>🚨 Emergency Steps</h4><ol style="padding-left:20px;">{steps_html}</ol></div>',
                unsafe_allow_html=True,
            )

        # Government Help
        links_html = "".join(
            f'<a class="resource-link" href="{res["link"]}" target="_blank">{res["icon"]} <div>'
            f'<div>{res["label"]}</div><div style="font-size:0.8rem;color:#666;">{res["value"]}</div></div></a>'
            for res in utils.GOVERNMENT_RESOURCES
        )
        st.markdown(
            f'<div class="light-card"><h4>🏛 Government Help & Resources</h4>{links_html}</div>',
            unsafe_allow_html=True,
        )

        # Safety Tip
        st.markdown(
            f'<div class="light-card" style="border-left-color:{utils.SEVERITY_COLORS["Low"]};">'
            f'<h4>🛡 Safety Tip</h4><p><i>{r["safety_tip"]}</i></p></div>',
            unsafe_allow_html=True,
        )

        # Download PDF
        pdf_bytes = utils.build_pdf_report(r)
        st.download_button(
            "⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"ScamLens_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )



# MODEL INSIGHTS PAGE

elif page == "📈 Model Insights":
    m = load_metrics()

    st.markdown("<h1 style='font-size:2.2rem;'>📈 Model Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB0D9;'>Decision Tree Classifier — Fraud Severity Prediction.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="num">{m["accuracy"]*100:.1f}%</div><div class="lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="num">{m["n_test"]}</div><div class="lbl">Test Samples</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="num">Decision Tree</div><div class="lbl">Algorithm (entropy, depth 5)</div></div>', unsafe_allow_html=True)

    st.markdown("###  ")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="light-card"><h4>📋 Classification Report</h4>', unsafe_allow_html=True)
        report_df = pd.DataFrame(m["report"]).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        cm_fig = px.imshow(
            m["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
            x=m["classes"], y=m["classes"],
            labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix",
        )
        st.plotly_chart(style_fig(cm_fig, height=380), use_container_width=True)



# INSIGHTS & CONCLUSION PAGE

else:
    st.markdown("<h1 style='font-size:2.2rem;'>📌 Insights & Conclusion</h1>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📊 Key Insights", "✅ Conclusion", "🚀 Future Scope"])

    with t1:
        insights = [
            "Fraud transactions are concentrated in specific fraud categories and time periods — targeted monitoring can improve detection efficiency.",
            "High-value UPI transactions carry greater risk, making real-time verification essential for secure digital payments.",
            "Certain UPI apps and fraud techniques appear more frequently in scam cases, highlighting the need for platform-specific security measures.",
            "The same rupee amount does not mean the same thing to every victim — income-adjusted Financial Impact reveals damage that raw severity hides.",
            "The Decision Tree model rediscovers the exact business-rule thresholds from raw data, demonstrating transparent, explainable classification.",
            "Recovery guidance (government helplines, recommended action) is what turns a severity label into something actually useful to a victim.",
        ]
        for i, ins in enumerate(insights, start=1):
            st.markdown(f'<div class="light-card" style="border-left-color:var(--teal);"><b>{i}.</b> {ins}</div>', unsafe_allow_html=True)

    with t2:
        st.markdown(
            '<div class="glass-card"><p>This project analyzed UPI fraud transaction data using Python, '
            'Data Science, and Machine Learning to identify fraud patterns and classify fraud severity.</p>'
            '<p>Beyond classification, the project deliberately separates what ML should do (learn the '
            'severity rule from transaction data) from what a simple calculation should do (financial '
            'impact, recommended action) — keeping the system both accurate and honest about how it works.</p>'
            '<p>The result, ScamLens AI, is framed as a <b>recovery assistant</b>: it doesn\'t just tell a '
            'victim "this was severe" after their money is already gone — it tells them what to do next.</p>'
            '</div>', unsafe_allow_html=True,
        )

    with t3:
        future = [
            "Apply advanced algorithms (Random Forest, XGBoost) as the dataset grows.",
            "Integrate real-time UPI transaction monitoring.",
            "Add an AI-based fraud risk score computed before a transaction completes.",
            "Add SMS/Email alerts for suspicious transactions.",
            "Build a dashboard view for banks and investigators.",
            "Deploy on Streamlit Cloud for public access.",
        ]
        for f in future:
            st.markdown(f"- {f}")
