#  NewsCrew AI — Real-Time News Summarizer

A production-ready AI application that summarizes real-time global news using a **Retrieval Augmented Generation (RAG)** pipeline orchestrated by **CrewAI**.

## Architecture

```
User Query → Streamlit UI → FastAPI Backend → CrewAI Crew
                                                    ↓
                              ┌─────────────────────────────────────┐
                              │  Agent 1: News Research Specialist  │
                              │  Tool: WebSearchTool (Tavily)       │
                              │         ↓ stores in ChromaDB        │
                              ├─────────────────────────────────────┤
                              │  Agent 2: News Analyst & Summarizer │
                              │  Tool: RAGRetrieveTool (ChromaDB)   │
                              │         ↓ sends to Gemini AI        │
                              └─────────────────────────────────────┘
                                                    ↓
                                         News Summary Output
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| AI Orchestration | CrewAI (Sequential Process) |
| RAG Framework | LangChain |
| Web Search | Tavily API |
| Vector Database | ChromaDB (local) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Google Gemini 1.5 Flash |

## Project Structure

```
news-crew-app/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── main.py             # FastAPI app & /summarize endpoint
│   ├── agent.py            # CrewAI agents, tasks & crew runner
│   ├── rag_chain.py        # Tavily search + ChromaDB RAG
│   └── gemini_llm.py       # Gemini API wrapper
├── data/                   # ChromaDB persistent storage
├── requirements.txt
├── .env                    # API keys (fill this in)
└── README.md
```

## Setup

### 1. Get API Keys

- **Tavily**: Sign up free at [tavily.com](https://tavily.com)
- **Gemini**: Get key at [aistudio.google.com](https://aistudio.google.com) 

### 2. Configure Environment

Edit `.env`:

```env
TAVILY_API_KEY=your_tavily_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: This will download the `all-MiniLM-L6-v2` embedding model (~80MB) on first run.

### 4. Run the Application

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run app.py
```



## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health + API key validation |
| `/summarize` | POST | Run the full RAG pipeline |
| `/docs` | GET | Interactive Swagger UI |

### POST /summarize

Request:
```json
{
  "query": "Latest AI breakthroughs 2024"
}
```

Response:
```json
{
  "status": "success",
  "query": "Latest AI breakthroughs 2024",
  "summary": "##  News Summary: ...",
  "duration_seconds": 45.2,
  "error": null
}
```

## Example Queries

- `Latest AI breakthroughs 2026`
- `World political crisis today`
- `New vaccine updates and healthcare news`
- `Latest technology changes in IT industry`
- `Global climate change updates`
- `Stock market news today`
- `Space exploration discoveries`
- `Cybersecurity threats 2026`

## How It Works

1. **Query** → User submits a news topic via the Streamlit UI
2. **FastAPI** → Validates and forwards the request to the CrewAI crew
3. **Agent 1 (Research Specialist)**:
   - Calls `WebSearchTool` with the query
   - Tavily fetches 8 recent news articles
   - Articles are chunked and embedded via `all-MiniLM-L6-v2`
   - Chunks stored in local ChromaDB vector store
4. **Agent 2 (Analyst & Summarizer)**:
   - Calls `RAGRetrieveTool` with the same query
   - ChromaDB returns top-10 most relevant chunks via cosine similarity
   - Context sent to Gemini 1.5 Flash with structured prompt
   - Gemini generates a formatted Markdown summary
5. **Output** → Summary returned through FastAPI → displayed in Streamlit

## Performance

- Typical pipeline time: **30–90 seconds** (depends on Tavily + Gemini latency)
- ChromaDB persists between runs — searches accumulate over time
- Embedding model is pre-warmed on backend startup



