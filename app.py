import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.analytics import load_and_preprocess_shipments, compute_weekly_route_aggregates, compute_baselines_and_flags
from src.assistant import ConversationalAssistant
from src.inference import LocalQwenEngine
from src.tracker import TokenCostTracker
from src.config import FreightConfig, DEFAULT_CONFIG

st.set_page_config(
    page_title="FreightTiger Shipping Cost Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-justified {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-flagged {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    shipments_df = load_and_preprocess_shipments(DEFAULT_CONFIG.shipments_path)
    weekly_df = compute_weekly_route_aggregates(shipments_df)
    all_metrics_df, _ = compute_baselines_and_flags(weekly_df, config=DEFAULT_CONFIG)
    
    if os.path.exists(DEFAULT_CONFIG.output_path):
        output_df = pd.read_csv(DEFAULT_CONFIG.output_path)
    else:
        output_df = pd.DataFrame()
        
    notes_df = pd.read_csv(DEFAULT_CONFIG.notes_path)
    
    if os.path.exists(DEFAULT_CONFIG.token_log_path):
        token_df = pd.read_csv(DEFAULT_CONFIG.token_log_path)
    else:
        token_df = pd.DataFrame()
        
    return shipments_df, all_metrics_df, output_df, notes_df, token_df

shipments_df, all_metrics_df, output_df, notes_df, token_df = load_data()

# Header
st.markdown('<div class="main-header">🚚 FreightTiger Smart Shipping Cost Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered weekly cost-per-tonne-km tracking, rolling baseline analysis, and context-grounded anomaly audits.</div>', unsafe_allow_html=True)

# Top KPI row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Shipments", f"{len(shipments_df):,}")
with col2:
    st.metric("Route-Weeks Audited", f"{len(all_metrics_df):,}")
with col3:
    spikes_cnt = len(output_df) if not output_df.empty else 0
    st.metric("Candidate Spikes", f"{spikes_cnt}")
with col4:
    justified_cnt = (output_df['flagged'] == 'No (justified)').sum() if not output_df.empty else 0
    st.metric("Justified Spikes", f"{justified_cnt}", delta="Grounded by Notes")
with col5:
    flagged_cnt = (output_df['flagged'] == 'Yes').sum() if not output_df.empty else 0
    st.metric("Flagged for Review", f"{flagged_cnt}", delta="-Requires Human Action", delta_color="inverse")

# Tabs for Main View, Route Visualizer, Conversational Assistant, and Observability
tab_anomalies, tab_visualizer, tab_chat, tab_observability = st.tabs([
    "📋 Anomaly Audit Table",
    "📈 Route Baseline Visualizer",
    "💬 Conversational Assistant (Stretch Goal)",
    "⚙️ Observability & Token Cost"
])

# ----------------- TAB 1: ANOMALIES -----------------
with tab_anomalies:
    st.subheader("Audited Candidate Spikes (Matching Official Output Contract)")
    
    # Filter controls
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        route_options = ["All Routes"] + sorted(list(output_df['route'].unique()))
        selected_route = st.selectbox("Filter by Route", route_options, key="audit_route")
    with f_col2:
        status_options = ["All Verdicts", "Flagged for Review (Yes)", "Justified (No)"]
        selected_status = st.selectbox("Filter by Verdict", status_options, key="audit_status")
        
    filtered_df = output_df.copy()
    if selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['route'] == selected_route]
    if selected_status == "Flagged for Review (Yes)":
        filtered_df = filtered_df[filtered_df['flagged'] == 'Yes']
    elif selected_status == "Justified (No)":
        filtered_df = filtered_df[filtered_df['flagged'] == 'No (justified)']
        
    st.dataframe(
        filtered_df,
        column_config={
            "route": st.column_config.TextColumn("Route", width="medium"),
            "week_of": st.column_config.TextColumn("Week Of", width="small"),
            "cost_per_tonne_km": st.column_config.NumberColumn("Cost / T-Km (₹)", format="%.2f"),
            "vs_own_history": st.column_config.TextColumn("vs. Own History"),
            "vs_similar_routes": st.column_config.TextColumn("vs. Similar Routes"),
            "flagged": st.column_config.TextColumn("Flagged Status"),
            "matched_note_id": st.column_config.TextColumn("Matched Note ID"),
            "reason": st.column_config.TextColumn("Reason", width="large")
        },
        use_container_width=True,
        hide_index=True
    )
    
    csv_bytes = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Output CSV (output.csv)",
        data=csv_bytes,
        file_name="output.csv",
        mime="text/csv"
    )

# ----------------- TAB 2: ROUTE VISUALIZER -----------------
with tab_visualizer:
    st.subheader("Weekly Cost Tracking vs. Rolling & Peer Baselines")
    
    # Calculate global range across all routes for standardized cross-route visual comparison
    global_cost_max = float(all_metrics_df['cost_per_tonne_km_raw'].max())
    default_y_max = round(global_cost_max * 1.12, 1)  # e.g., 5.0
    
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        v_routes = sorted(list(all_metrics_df['route'].unique()))
        target_route = st.selectbox("Select Route to Drill Down", v_routes, index=0)
    with ctrl_col2:
        axis_mode = st.radio(
            "Y-Axis (Cost / T-Km) Scale",
            [f"Standard Global Scale (₹0 to ₹{default_y_max:.2f}) [Comparable across routes]",
             f"Zoomed Scale (₹1.50 to ₹{default_y_max:.2f})",
             "Dynamic Auto-Scale (Route-specific)"],
            index=0,
            horizontal=False
        )
        
    r_data = all_metrics_df[all_metrics_df['route'] == target_route].sort_values('week_of')
    
    fig = go.Figure()
    # Actual cost line
    fig.add_trace(go.Scatter(
        x=r_data['week_of'],
        y=r_data['cost_per_tonne_km_raw'],
        mode='lines+markers',
        name='Actual Cost / Tonne-Km',
        line=dict(color='#2563EB', width=2.5)
    ))
    # 8-week baseline
    fig.add_trace(go.Scatter(
        x=r_data['week_of'],
        y=r_data['hist_avg'],
        mode='lines',
        name='Trailing 8-Week Baseline',
        line=dict(color='#10B981', dash='dash', width=2)
    ))
    # Peer baseline
    fig.add_trace(go.Scatter(
        x=r_data['week_of'],
        y=r_data['peer_avg'],
        mode='lines',
        name='Peer Routes Baseline (Same Type)',
        line=dict(color='#F59E0B', dash='dot', width=2)
    ))
    
    # Mark anomalies
    r_spikes = r_data[r_data['is_candidate_spike']]
    if not r_spikes.empty:
        fig.add_trace(go.Scatter(
            x=r_spikes['week_of'],
            y=r_spikes['cost_per_tonne_km_raw'],
            mode='markers',
            name='Candidate Cost Spike',
            marker=dict(color='#EF4444', size=11, symbol='circle-open-dot', line=dict(width=3))
        ))
        
    # Configure Y-axis range based on user selection
    yaxis_config = dict(
        title="Cost per Tonne-Km (₹)",
        gridcolor="#E2E8F0"
    )
    if "Standard Global Scale" in axis_mode:
        yaxis_config["range"] = [0.0, default_y_max]
    elif "Zoomed Scale" in axis_mode:
        yaxis_config["range"] = [1.5, default_y_max]
        
    fig.update_layout(
        title=f"Cost Dynamics & Baselines for {target_route}",
        xaxis_title="Week (Monday Date)",
        yaxis=yaxis_config,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-Route Overlay Comparison
    with st.expander("📊 Compare Multiple Corridors on the Same Fixed Scale"):
        st.write("Select routes to overlay and compare their actual cost trajectories side-by-side on the exact same axis.")
        multi_selected = st.multiselect(
            "Select Corridors to Overlay",
            v_routes,
            default=v_routes[:3] if len(v_routes) >= 3 else v_routes
        )
        if multi_selected:
            multi_fig = go.Figure()
            for r in multi_selected:
                m_df = all_metrics_df[all_metrics_df['route'] == r].sort_values('week_of')
                multi_fig.add_trace(go.Scatter(
                    x=m_df['week_of'],
                    y=m_df['cost_per_tonne_km_raw'],
                    mode='lines',
                    name=r,
                    line=dict(width=2)
                ))
            multi_fig.update_layout(
                title="Cross-Corridor Cost Comparison (Shared Axis)",
                xaxis_title="Week (Monday Date)",
                yaxis=dict(title="Cost per Tonne-Km (₹)", range=[0.0, default_y_max]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(multi_fig, use_container_width=True)

@st.cache_resource
def get_inference_engine():
    tracker = TokenCostTracker(model_name=DEFAULT_CONFIG.model_name)
    return LocalQwenEngine(config=DEFAULT_CONFIG, tracker=tracker)

def get_conversational_assistant():
    engine = get_inference_engine()
    return ConversationalAssistant(
        engine=engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=engine.tracker,
        config=DEFAULT_CONFIG
    )



# ----------------- TAB 3: SMART ASSISTANT -----------------
with tab_chat:
    st.subheader("Smart Shipping Cost Assistant")
    st.write("Ask questions in plain English about route cost changes, check whether cost rises were justified by operational context notes, or audit historical baselines.")

    assistant = get_conversational_assistant()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your FreightTiger Smart Shipping Cost Assistant.**\n\n"
                    "Ask me about any shipping route cost rise (e.g. *'Why was Ahmedabad-Mumbai high on 2025-01-20?'*), "
                    "check normal baseline weeks, or investigate unexplained rises."
                )
            }
        ]

    # Render conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "caption" in msg and msg["caption"]:
                st.caption(msg["caption"])

    # Example query quick chips
    st.markdown("**Suggested Inquiries:**")
    c1, c2, c3, c4 = st.columns(4)
    selected_query = None
    with c1:
        if st.button("Ahmedabad-Mumbai (Jan 2025)", use_container_width=True):
            selected_query = "Why was Ahmedabad-Mumbai high on 2025-01-20?"
    with c2:
        if st.button("Chennai-Bangalore (Feb 2025)", use_container_width=True):
            selected_query = "Why did Chennai-Bangalore spike on 2025-02-24?"
    with c3:
        if st.button("Delhi-Jaipur (Nov 2024)", use_container_width=True):
            selected_query = "Why was Delhi-Jaipur flagged on 2024-11-11?"
    with c4:
        if st.button("Mumbai-Delhi baseline (Jul 2024)", use_container_width=True):
            selected_query = "Why did Mumbai-Delhi cost 2.06 on 2024-07-29?"

    chat_input_val = st.chat_input("Ask a question about shipping costs (e.g., Why was Ahmedabad-Mumbai high in Jan 2025?)...")
    active_query = chat_input_val or selected_query

    if active_query:
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.spinner("Analyzing verified anomaly records and baselines..."):
            res = assistant.answer(active_query)

        ans_text = res["answer"]
        caption = f"⏱️ Latency: {res['latency_seconds']:.2f}s | Status: `{res['status']}`"
        if res.get("parsed_params") and res["parsed_params"].get("resolved_week_of"):
            caption += f" | Evaluated Week: `{res['parsed_params']['resolved_week_of']}`"

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ans_text,
            "caption": caption
        })

        with st.chat_message("assistant"):
            st.markdown(ans_text)
            st.caption(caption)

# ----------------- TAB 4: OBSERVABILITY -----------------
with tab_observability:
    st.subheader("LLM Observability & Token Cost Metrics")
    st.write("Detailed audit trail of all model inference requests, token counts, and cost logs.")
    
    if not token_df.empty:
        obs_col1, obs_col2, obs_col3, obs_col4 = st.columns(4)
        with obs_col1:
            st.metric("Total Model Calls", f"{len(token_df)}")
        with obs_col2:
            st.metric("Total Input Tokens", f"{token_df['input_tokens'].sum():,}")
        with obs_col3:
            st.metric("Total Output Tokens", f"{token_df['output_tokens'].sum():,}")
        with obs_col4:
            st.metric("Cloud Cost Equivalent", f"${token_df['cost_usd'].sum():.6f}")
            
        st.dataframe(token_df, use_container_width=True)
    else:
        st.write("No token log found. Run pipeline.py to generate token usage logs.")
