import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time

# Set up page configurations
st.set_page_config(
    page_title="EmbedShield | Semantic Guardrail & Entropy Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint URL (Localhost inside the single Docker container)
API_URL = "http://localhost:8000"

# Custom CSS for rich aesthetics and dark-mode styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Metrics Row Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.1), 0 4px 6px -4px rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .metric-title {
        color: #475569;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .badge {
        padding: 6px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-width: 1px;
        border-style: solid;
    }
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.1);
        color: #065f46;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .badge-block {
        background-color: rgba(239, 68, 68, 0.1);
        color: #991b1b;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(90deg, #e0e7ff 0%, #f1f5f9 100%);
        backdrop-filter: blur(15px);
        border-bottom: 2px solid #6366f1;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Threat Table */
    .threat-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .threat-table th {
        background-color: #f1f5f9;
        color: #475569;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        padding: 10px;
        border-bottom: 1px solid #e2e8f0;
        text-align: left;
    }
    .threat-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #334155;
    }
    
    /* Live Log Card */
    .log-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-family: monospace;
        font-size: 0.75rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to make requests to the FastAPI backend
def call_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except Exception:
        return False, {}

def call_api_safe_prompts():
    try:
        response = requests.get(f"{API_URL}/api/safe-prompts", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching safe prompts from API: {e}")
    return []

def call_api_shield(payload):
    try:
        response = requests.post(f"{API_URL}/api/shield", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error sending check request to API: {e}")
    return None

# Page Title & Glassmorphic Header
st.markdown("""
<div class="header-card">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="color: #1e1b4b; margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em;">🛡️ EmbedShield</h1>
            <p style="color: #475569; margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 400;">
                Unsupervised Semantic Density & Shannon Entropy Guardrail Gateway for Agentic LLM Inputs
            </p>
        </div>
        <div style="text-align: right;">
            <span style="color: #4f46e5; font-weight: 600; font-size: 0.85rem; padding: 6px 12px; border-radius: 20px; background: #e0e7ff; border: 1px solid #c7d2fe;">
                Running on Intel CPU
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State for tracking history
if "history" not in st.session_state:
    st.session_state.history = []

# Fetch health and connect to API
api_available, health_data = call_api_health()

if not api_available:
    st.warning("⚠️ API Gateway at `http://localhost:8000` is currently unreachable. Make sure the Docker container or local FastAPI backend is running.")
    st.stop()

# Load safe training dataset
safe_data = call_api_safe_prompts()
if not safe_data:
    st.error("No safe training data received from the API gateway.")
    st.stop()

# Sidebar: Controls & Parameter Tuning
st.sidebar.markdown("### ⚙️ Boundary Configuration")

# Selection of method
method = st.sidebar.selectbox(
    "Semantic Clustering Algorithm",
    options=["LOF", "DBSCAN"],
    help="LOF calculates local density deviation. DBSCAN forms tight density-based core clusters."
)

if method == "LOF":
    st.sidebar.markdown("**LOF Hyperparameters**")
    lof_contamination = st.sidebar.slider(
        "Contamination Rate",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Proportion of outliers in the training dataset (usually 5% to 15%)"
    )
    lof_neighbors = st.sidebar.slider(
        "Number of Neighbors",
        min_value=3,
        max_value=30,
        value=15,
        step=1,
        help="Higher values average local density over a wider radius"
    )
    dbscan_eps = 0.45
    dbscan_min_samples = 3
else:
    st.sidebar.markdown("**DBSCAN Hyperparameters**")
    dbscan_eps = st.sidebar.slider(
        "Epsilon (Radius)",
        min_value=0.10,
        max_value=1.50,
        value=0.45,
        step=0.05,
        help="Max L2 distance between two samples to be considered in the same neighborhood"
    )
    dbscan_min_samples = st.sidebar.slider(
        "Min Core Samples",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Min samples in epsilon radius to declare a cluster core point"
    )
    lof_contamination = 0.10
    lof_neighbors = 15

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌀 Entropy Radar Bounds")
st.sidebar.markdown("Formula: $H(X) = -\\sum P(x) \\log_2 P(x)$")

entropy_min = st.sidebar.slider(
    "Min Entropy Threshold",
    min_value=0.0,
    max_value=4.5,
    value=3.5,
    step=0.1,
    help="Flags inputs that are highly ordered/repetitive (e.g. DDoS fuzzing)"
)

entropy_max = st.sidebar.slider(
    "Max Entropy Threshold",
    min_value=4.0,
    max_value=8.0,
    value=4.8,
    step=0.1,
    help="Flags inputs with chaotic randomness (e.g. obfuscated Base64, hex payloads)"
)

# Statistics & Analytics Dashboard (Calculated from History)
history_df = pd.DataFrame(st.session_state.history)

total_scanned = len(st.session_state.history)
passed_cnt = len(history_df[history_df["status"] == "PASS"]) if total_scanned > 0 else 0
blocked_semantic = len(history_df[history_df["is_semantic_outlier"] == True]) if total_scanned > 0 else 0
blocked_entropy = len(history_df[history_df["is_entropy_outlier"] == True]) if total_scanned > 0 else 0
block_rate = (total_scanned - passed_cnt) / total_scanned * 100 if total_scanned > 0 else 0.0

# Render Statistics Cards
cols = st.columns(5)
with cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total API Scans</div>
        <div class="metric-value">{total_scanned}</div>
        <div class="metric-sub" style="color: #475569;">Active Session Logs</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Passed to LLM</div>
        <div class="metric-value" style="color: #059669;">{passed_cnt}</div>
        <div class="metric-sub" style="color: #059669;">✅ Safe Inliers</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Blocked Semantic</div>
        <div class="metric-value" style="color: #dc2626;">{blocked_semantic}</div>
        <div class="metric-sub" style="color: #dc2626;">❌ Off-Topic Outliers</div>
    </div>
    """, unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Blocked Entropy</div>
        <div class="metric-value" style="color: #d97706;">{blocked_entropy}</div>
        <div class="metric-sub" style="color: #d97706;">⚠️ Structured/Chaos</div>
    </div>
    """, unsafe_allow_html=True)
with cols[4]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Block Rate</div>
        <div class="metric-value" style="color: #db2777;">{block_rate:.1f}%</div>
        <div class="metric-sub" style="color: #db2777;">🛡️ Mitigated Risk</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 1. Live Input Terminal
st.markdown("### 🔌 Real-Time Prompt Ingest Gateway")
with st.container():
    col_input, col_btn = st.columns([10, 2])
    with col_input:
        user_input = st.text_input(
            "Enter raw user prompt to pass through gate:",
            placeholder="Type normal text, base64 payload, or highly repetitive spam to inspect boundaries...",
            label_visibility="collapsed"
        )
    with col_btn:
        scan_triggered = st.button("🚀 Screen Input", use_container_width=True)

# Action when user submits a prompt
if (scan_triggered or user_input) and user_input.strip() != "":
    # Prepare payload with current sidebar configurations
    payload = {
        "prompt": user_input,
        "method": method,
        "lof_contamination": lof_contamination,
        "lof_neighbors": lof_neighbors,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max
    }
    
    with st.spinner("Screening prompt..."):
        result = call_api_shield(payload)
        
    if result:
        # Check if prompt already exists in history to avoid duplication on refresh
        exists = any(h["prompt"] == result["prompt"] for h in st.session_state.history)
        if not exists:
            st.session_state.history.insert(0, result) # Insert at front
            st.rerun()

# Layout splits: Visual Cluster Space and Decision Engine Logs
col_left, col_right = st.columns([7, 5])

# Left column: Interactive 2D Plotly Scatter Plot
with col_left:
    st.markdown("### 🗺️ Unsupervised Semantic Cluster Boundaries (PCA)")
    
    # Render Plotly Map
    fig = go.Figure()
    
    # Group safe prompts by category to style them differently
    safe_df = pd.DataFrame(safe_data)
    
    # Custom color palette for categories
    category_colors = {
        "Customer Support": "#059669",  # Emerald Green
        "Technical & Coding": "#2563eb", # Royal Blue
        "General & Greetings": "#7c3aed" # Deep Purple
    }
    
    # 1. Add Safe Dataset traces
    for category, df_cat in safe_df.groupby("category"):
        fig.add_trace(go.Scatter(
            x=df_cat["x"],
            y=df_cat["y"],
            mode="markers",
            marker=dict(
                size=10,
                color=category_colors.get(category, "#94a3b8"),
                opacity=0.6,
                line=dict(width=1, color="rgba(0,0,0,0.1)")
            ),
            name=f"Safe: {category}",
            text=df_cat["prompt"].apply(lambda t: t[:40] + ("..." if len(t) > 40 else "")),
            hoverinfo="text"
        ))
        
    # 2. Add history points if they exist
    if len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        
        # Split history points into categories for visual plotting
        # Category A: Passed points (Inliers & normal entropy)
        passed_hist = hist_df[(hist_df["status"] == "PASS")]
        if not passed_hist.empty:
            fig.add_trace(go.Scatter(
                x=passed_hist["x"],
                y=passed_hist["y"],
                mode="markers",
                marker=dict(
                    size=12,
                    color="#059669", # Emerald Green
                    symbol="circle",
                    line=dict(width=2, color="#0f172a")
                ),
                name="Passed Prompts (Inliers)",
                text=passed_hist["prompt"].apply(lambda t: f"<b>PASS</b><br>{t[:50]}..."),
                hoverinfo="text"
            ))
            
        # Category B: Semantic Outliers (Out of cluster density)
        sem_outliers = hist_df[(hist_df["is_semantic_outlier"] == True)]
        if not sem_outliers.empty:
            fig.add_trace(go.Scatter(
                x=sem_outliers["x"],
                y=sem_outliers["y"],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#dc2626", # Red
                    symbol="x",
                    line=dict(width=2, color="#dc2626")
                ),
                name="Semantic Outliers (Off-Topic)",
                text=sem_outliers["prompt"].apply(lambda t: f"<b>BLOCK: Semantic</b><br>{t[:50]}..."),
                hoverinfo="text"
            ))
            
        # Category C: Entropy Anomalies (Ordered repetitions or obfuscated high-entropy)
        entropy_outliers = hist_df[(hist_df["is_entropy_outlier"] == True)]
        if not entropy_outliers.empty:
            fig.add_trace(go.Scatter(
                x=entropy_outliers["x"],
                y=entropy_outliers["y"],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#d97706", # Amber
                    symbol="triangle-up",
                    line=dict(width=2, color="#0f172a")
                ),
                name="Entropy Danger Zone",
                text=entropy_outliers["prompt"].apply(lambda t: f"<b>BLOCK: Entropy Anomaly</b><br>{t[:50]}..."),
                hoverinfo="text"
            ))
            
        # 3. Highlight the most recent scanned point
        latest_scan = st.session_state.history[0]
        fig.add_trace(go.Scatter(
            x=[latest_scan["x"]],
            y=[latest_scan["y"]],
            mode="markers+text",
            marker=dict(
                size=22,
                color="#4f46e5", # Indigo
                symbol="star",
                line=dict(width=2, color="#0f172a")
            ),
            name="🔴 LATEST INGESTION",
            text=["LATEST"],
            textposition="top center",
            textfont=dict(color="#0f172a", size=10, family="Inter"),
            hoverinfo="none"
        ))

    # Layout styling for the Graph
    fig.update_layout(
        template="plotly",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, tickfont=dict(color="#475569")),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, tickfont=dict(color="#475569")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569")
        ),
        height=520
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Right column: Logs, Details, Table
with col_right:
    # A. Active Inspection Panel
    st.markdown("### 🔍 Live Payload Inspection Side-Panel")
    
    if len(st.session_state.history) > 0:
        latest = st.session_state.history[0]
        badge_style = "badge-pass" if latest["status"] == "PASS" else "badge-block"
        
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; color: #475569; font-size: 0.8rem; text-transform: uppercase;">Engine Decision</span>
                <span class="badge {badge_style}">{latest["status"]}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="font-weight: 600; font-size: 0.85rem; color: #475569;">Prompt Content:</span>
                <p style="font-size: 0.9rem; color: #0f172a; font-style: italic; margin-top: 4px; line-height: 1.4; max-height: 100px; overflow-y: auto;">
                    "{latest["prompt"]}"
                </p>
            </div>
            <div style="display: flex; gap: 20px; border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 12px;">
                <div>
                    <span style="display: block; font-size: 0.75rem; color: #475569; text-transform: uppercase;">Entropy (H)</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #0284c7;">{latest["entropy"]:.3f}</span>
                </div>
                <div>
                    <span style="display: block; font-size: 0.75rem; color: #475569; text-transform: uppercase;">Semantic Outlier</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: {'#dc2626' if latest['is_semantic_outlier'] else '#059669'};">
                        {str(latest['is_semantic_outlier'])}
                    </span>
                </div>
                <div>
                    <span style="display: block; font-size: 0.75rem; color: #475569; text-transform: uppercase;">2D Projection</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #7c3aed;">({latest['x']:.2f}, {latest['y']:.2f})</span>
                </div>
            </div>
            <div style="margin-top: 15px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; border-left: 4px solid {'#059669' if latest['status'] == 'PASS' else '#dc2626'};">
                <span style="font-weight: 600; font-size: 0.75rem; color: #475569; display: block; text-transform: uppercase; margin-bottom: 2px;">Diagnostics Log:</span>
                <span style="font-size: 0.85rem; color: #0f172a; font-family: monospace;">{latest["reason"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 Ingest a user prompt above to see live diagnostic reports and entropy score validation.")

    # B. How Entropy Unmasks Hidden Threats (User Requested Static Demonstration Table)
    st.markdown("<br>### ⚡ How Entropy Unmasks Obfuscation (The Power of Chaos)", unsafe_allow_html=True)
    st.markdown("""
    <table class="threat-table">
        <thead>
            <tr>
                <th>Prompt Input Sample</th>
                <th>Semantic Group</th>
                <th>Entropy (H)</th>
                <th>Action</th>
                <th>Why Entropy Caught It</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>"How do I reset my password?"</b></td>
                <td>Inside Safe Cluster</td>
                <td><span style="color: #0284c7; font-weight: 600;">4.1 (Normal)</span></td>
                <td><span class="badge badge-pass">✅ Pass</span></td>
                <td>Standard, highly predictable character frequency of natural English prose.</td>
            </tr>
            <tr>
                <td><b>"A A A A A A A A A A A A A A"</b></td>
                <td>Edge of Safe Cluster</td>
                <td><span style="color: #d97706; font-weight: 600;">0.0 (Ultra-Low)</span></td>
                <td><span class="badge badge-block">❌ Block</span></td>
                <td>Zero randomness. Repetitive tokens, typical of context-window overloading attacks.</td>
            </tr>
            <tr>
                <td><b>"U2VjdXJpdHkgYnJlYWNoIHRlc3Q..."</b> (Base64)</td>
                <td>Mapped near Safe Zone</td>
                <td><span style="color: #dc2626; font-weight: 600;">5.9 (Ultra-High)</span></td>
                <td><span class="badge badge-block">❌ Block</span></td>
                <td>High character variety. Model embedding was confused, but Entropy caught the chaos.</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

# Bottom section: Session History Logs
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### 🕒 Active Gateway Session Logs")
    
    # Display clear button
    if st.button("🗑️ Clear Local History"):
        st.session_state.history = []
        st.rerun()
        
    for index, h in enumerate(st.session_state.history):
        badge_color = "#059669" if h["status"] == "PASS" else "#dc2626"
        st.markdown(f"""
        <div class="log-card">
            <span style="color: {badge_color}; font-weight: bold;">[{h["status"]}]</span> 
            <b>Entropy:</b> {h["entropy"]:.3f} | 
            <b>Semantic Outlier:</b> {str(h["is_semantic_outlier"])} | 
            <b>Coordinates:</b> ({h["x"]:.3f}, {h["y"]:.3f}) <br>
            <span style="color: #475569; font-weight: 600;">Prompt:</span> <span style="color: #0f172a;">"{h["prompt"]}"</span> <br>
            <span style="color: #475569; font-weight: 600;">Reason:</span> <span style="color: #334155; font-style: italic;">{h["reason"]}</span>
        </div>
        """, unsafe_allow_html=True)
