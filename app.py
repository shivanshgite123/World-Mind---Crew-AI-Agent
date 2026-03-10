"""
News Crew Summarizer — Streamlit Frontend

A production-grade UI for the AI-powered news summarization pipeline.
Design: Editorial dark theme with amber accents, inspired by financial terminals.
"""

import streamlit as st
import requests
import time
from datetime import datetime

#  Page Config 
st.set_page_config(
    page_title="NewsCrew AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Styling 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Global reset */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark terminal background */
    .stApp {
        background-color: #0a0c10;
        color: #e2e8f0;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid #1e2533;
        margin-bottom: 2.5rem;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        font-weight: 400;
        color: #f0e6d3;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.1;
    }

    .hero-title span {
        color: #f59e0b;
        font-style: italic;
    }

    .hero-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.75rem;
    }

    .ticker-bar {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #f59e0b;
        background: #111827;
        border: 1px solid #1e2533;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }

    /* Search box */
    .stTextInput > div > div > input {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #f0e6d3 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.05rem !important;
        padding: 0.85rem 1.1rem !important;
        transition: border-color 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #d97706, #f59e0b) !important;
        color: #0a0c10 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #b45309, #d97706) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.25) !important;
    }

    /* Pipeline status cards */
    .pipeline-card {
        background: #111827;
        border: 1px solid #1e2533;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 0.5rem 0;
    }

    .pipeline-card.active {
        border-color: #f59e0b;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.1);
    }

    .pipeline-card.done {
        border-color: #10b981;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.08);
    }

    .pipeline-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.3rem;
    }

    .pipeline-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #f0e6d3;
    }

    /* Result section */
    .result-wrapper {
        background: #0f1724;
        border: 1px solid #1e2533;
        border-radius: 12px;
        padding: 2rem 2.2rem;
        margin-top: 2rem;
    }

    .result-meta {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #64748b;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e2533;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Markdown inside result */
    .result-wrapper h2 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.6rem !important;
        color: #f0e6d3 !important;
        margin-bottom: 1rem !important;
    }

    .result-wrapper h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #f59e0b !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    .result-wrapper li {
        color: #cbd5e1 !important;
        line-height: 1.7 !important;
        margin-bottom: 0.3rem !important;
    }

    .result-wrapper p {
        color: #94a3b8 !important;
        line-height: 1.75 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e2533 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #64748b !important;
    }

    /* Query chips */
    .query-chip {
        display: inline-block;
        background: #1e2533;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #94a3b8;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    /* Metric boxes */
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-box {
        background: #111827;
        border: 1px solid #1e2533;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        color: #f59e0b;
        font-weight: 700;
    }

    .metric-lbl {
        font-size: 0.72rem;
        color: #4b5563;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Error box */
    .error-box {
        background: #1c0a0a;
        border: 1px solid #7f1d1d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        color: #fca5a5;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

#  Constants 
BACKEND_URL = "http://localhost:8000"

EXAMPLE_QUERIES = [
    "Latest AI breakthroughs 2024",
    "World political crisis today",
    "New vaccine updates healthcare",
    "Technology changes IT industry",
    "Global climate change updates",
    "Stock market news today",
    "Space exploration discoveries",
    "Cybersecurity threats 2024",
]

# Session State 
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


#  Sidebar 
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem; 
                    letter-spacing: 3px; color: #4b5563; text-transform: uppercase; 
                    margin-bottom: 0.5rem;">Architecture</div>
        <div style="font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #f0e6d3;">
            RAG Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin: 1.5rem 0; font-size: 0.82rem; color: #64748b; line-height: 1.8;">
        <div style="color: #f59e0b; font-family: 'Space Mono', monospace; font-size: 0.68rem; 
                    letter-spacing: 1px; margin-bottom: 0.6rem;">FLOW</div>
        <div> User Query</div>
        <div style="color: #2d3748; margin: 0 0 0 0.5rem;">│</div>
        <div> Tavily Web Search</div>
        <div style="color: #2d3748; margin: 0 0 0 0.5rem;">│</div>
        <div> ChromaDB Storage</div>
        <div style="color: #2d3748; margin: 0 0 0 0.5rem;">│</div>
        <div> Semantic Retrieval</div>
        <div style="color: #2d3748; margin: 0 0 0 0.5rem;">│</div>
        <div> Gemini Generation</div>
        <div style="color: #2d3748; margin: 0 0 0 0.5rem;">│</div>
        <div> Summary Output</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem; 
                letter-spacing: 3px; color: #4b5563; text-transform: uppercase; 
                margin-bottom: 0.8rem;">Tech Stack</div>
    """, unsafe_allow_html=True)

    tech = [
        ( "CrewAI", "Orchestration"),
        ( "Tavily", "Web Search"),
        ( "ChromaDB", "Vector Store"),
        ( "Gemini 1.5", "Generation"),
        ( "MiniLM-L6", "Embeddings"),
        ( "FastAPI", "Backend"),
    ]
    for icon, name, role in tech:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; 
                    padding: 0.35rem 0; border-bottom: 1px solid #1a2030;
                    font-size: 0.82rem;">
            <span>{icon} <span style="color: #e2e8f0;">{name}</span></span>
            <span style="color: #4b5563; font-family: 'Space Mono', monospace; 
                         font-size: 0.68rem;">{role}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # History
    if st.session_state.history:
        st.markdown("""
        <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem; 
                    letter-spacing: 3px; color: #4b5563; text-transform: uppercase; 
                    margin-bottom: 0.8rem;">Recent Queries</div>
        """, unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
            <div style="padding: 0.4rem 0.6rem; margin: 0.25rem 0; 
                        background: #111827; border-radius: 4px;
                        font-size: 0.78rem; color: #64748b;
                        border-left: 2px solid #374151;">
                {item['query'][:40]}{'...' if len(item['query']) > 40 else ''}
            </div>
            """, unsafe_allow_html=True)


# Main Content 

# Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">NewsCrew <span>AI</span></div>
    <div class="hero-subtitle">Real-Time News Intelligence · RAG Pipeline · CrewAI Orchestration</div>
</div>
""", unsafe_allow_html=True)

# Ticker
now = datetime.now().strftime("%H:%M:%S UTC")
st.markdown(f"""
<div class="ticker-bar">
    ● LIVE  &nbsp;|&nbsp; {now}  &nbsp;|&nbsp; 
    AGENTS: RESEARCH SPECIALIST + NEWS ANALYST  &nbsp;|&nbsp; 
    VECTOR DB: CHROMADB  &nbsp;|&nbsp;  LLM: GEMINI 1.5 FLASH
</div>
""", unsafe_allow_html=True)

#  Search Interface 
col_input, col_btn = st.columns([4, 1])

with col_input:
    query = st.text_input(
        label="query",
        placeholder="Enter a news topic — e.g. 'Latest AI breakthroughs 2024'",
        label_visibility="collapsed",
        key="main_query",
    )

with col_btn:
    search_clicked = st.button("▶ ANALYZE", use_container_width=True)

# Example queries
st.markdown("<div style='margin-top: 0.75rem;'>", unsafe_allow_html=True)
cols = st.columns(4)
for i, example in enumerate(EXAMPLE_QUERIES):
    with cols[i % 4]:
        if st.button(example, key=f"ex_{i}", use_container_width=True,
                     help=f"Search: {example}"):
            query = example
            search_clicked = True
st.markdown("</div>", unsafe_allow_html=True)

# Pipeline Execution 
if search_clicked and query and query.strip():
    query = query.strip()

    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)

    # Pipeline status display
    col1, col2 = st.columns(2)

    with col1:
        agent1_placeholder = st.empty()
        agent1_placeholder.markdown("""
        <div class="pipeline-card active">
            <div class="pipeline-label">Agent 1 · Running</div>
            <div class="pipeline-title"> News Research Specialist</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.4rem;">
                Searching Tavily → Ingesting to ChromaDB...
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        agent2_placeholder = st.empty()
        agent2_placeholder.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-label">Agent 2 · Waiting</div>
            <div class="pipeline-title"> News Analyst & Summarizer</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.4rem;">
                Waiting for research to complete...
            </div>
        </div>
        """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Initializing crew...")
    status_text = st.empty()

    # Simulate progress while waiting for API
    start_time = time.time()

    # Non-blocking progress animation
    for pct in range(0, 35, 5):
        progress_bar.progress(pct, text=f"Agent 1: Searching web for '{query}'...")
        time.sleep(0.3)

    # Make API call
    try:
        response = requests.post(
            f"{BACKEND_URL}/summarize",
            json={"query": query},
            timeout=180,
        )

        # Advance progress
        for pct in range(35, 70, 5):
            progress_bar.progress(pct, text="Agent 1: Storing articles in ChromaDB...")
            time.sleep(0.2)

        agent1_placeholder.markdown("""
        <div class="pipeline-card done">
            <div class="pipeline-label">Agent 1 · Complete ✓</div>
            <div class="pipeline-title"> News Research Specialist</div>
            <div style="font-size:0.8rem; color:#10b981; margin-top:0.4rem;">
                Articles retrieved and stored in ChromaDB
            </div>
        </div>
        """, unsafe_allow_html=True)

        agent2_placeholder.markdown("""
        <div class="pipeline-card active">
            <div class="pipeline-label">Agent 2 · Running</div>
            <div class="pipeline-title"> News Analyst & Summarizer</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.4rem;">
                Retrieving chunks → Generating summary via Gemini...
            </div>
        </div>
        """, unsafe_allow_html=True)

        for pct in range(70, 98, 5):
            progress_bar.progress(pct, text="Agent 2: Generating summary with Gemini AI...")
            time.sleep(0.2)

        duration = round(time.time() - start_time, 1)

        if response.status_code == 200:
            data = response.json()

            progress_bar.progress(100, text=" Pipeline complete!")
            agent2_placeholder.markdown("""
            <div class="pipeline-card done">
                <div class="pipeline-label">Agent 2 · Complete ✓</div>
                <div class="pipeline-title"> News Analyst & Summarizer</div>
                <div style="font-size:0.8rem; color:#10b981; margin-top:0.4rem;">
                    Summary generated successfully
                </div>
            </div>
            """, unsafe_allow_html=True)

            if data.get("status") == "success":
                summary = data.get("summary", "")
                api_duration = data.get("duration_seconds", duration)

                # Add to history
                st.session_state.history.append({
                    "query": query,
                    "summary": summary,
                    "duration": api_duration,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })
                st.session_state.last_result = {
                    "query": query,
                    "summary": summary,
                    "duration": api_duration,
                }

                # Metrics row
                word_count = len(summary.split())
                st.markdown(f"""
                <div class="metric-row" style="margin-top: 1.5rem;">
                    <div class="metric-box">
                        <div class="metric-val">{api_duration}s</div>
                        <div class="metric-lbl">Pipeline Time</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{word_count}</div>
                        <div class="metric-lbl">Words Generated</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">2</div>
                        <div class="metric-lbl">Agents Used</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Result display
                st.markdown(f"""
                <div class="result-meta">
                    Query: {query} &nbsp;|&nbsp; 
                    Generated: {datetime.now().strftime("%d %b %Y, %H:%M")} &nbsp;|&nbsp; 
                    Pipeline: CrewAI → Tavily → ChromaDB → Gemini
                </div>
                """, unsafe_allow_html=True)

                st.markdown(summary)

                # Download
                st.download_button(
                    label="⬇ Download Summary (Markdown)",
                    data=f"# News Summary: {query}\n\n{summary}",
                    file_name=f"news_summary_{query[:30].replace(' ', '_')}.md",
                    mime="text/markdown",
                )

            else:
                error_msg = data.get("error", "Unknown error")
                st.markdown(f"""
                <div class="error-box">
                     Pipeline Error<br><br>{error_msg}
                </div>
                """, unsafe_allow_html=True)

        else:
            error_detail = ""
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text

            progress_bar.progress(100, text="❌ Request failed")
            st.markdown(f"""
            <div class="error-box">
                 API Error (HTTP {response.status_code})<br><br>
                {error_detail}
            </div>
            """, unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
        progress_bar.progress(100, text=" Connection failed")
        st.markdown("""
        <div class="error-box">
             Cannot connect to backend (http://localhost:8000)<br><br>
            Make sure the FastAPI server is running:<br>
            <code>cd backend && uvicorn main:app --reload --port 8000</code>
        </div>
        """, unsafe_allow_html=True)

    except requests.exceptions.Timeout:
        progress_bar.progress(100, text=" Request timed out")
        st.markdown("""
        <div class="error-box">
            ⏱ Request timed out (>3 minutes)<br><br>
            The pipeline may still be running. Try a more specific query or check backend logs.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        progress_bar.progress(100, text=" Unexpected error")
        st.markdown(f"""
        <div class="error-box">
             Unexpected Error<br><br>{str(e)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif not st.session_state.last_result:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #2d3748;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📡</div>
        <div style="font-family: 'DM Serif Display', serif; font-size: 1.4rem; 
                    color: #374151; margin-bottom: 0.5rem;">
            Ready to Analyze
        </div>
        <div style="font-family: 'Space Mono', monospace; font-size: 0.72rem; 
                    color: #374151; letter-spacing: 1px;">
            Enter a news topic above or click an example query to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
