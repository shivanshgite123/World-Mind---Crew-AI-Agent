"""
RAG Pipeline: Tavily Web Search → ChromaDB Vector Store → Gemini Generation

Flow:
  1. WebSearchTool  → calls Tavily, embeds results, stores in ChromaDB
  2. RAGRetrieveTool → retrieves relevant chunks from ChromaDB, sends to Gemini
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from crewai.tools import BaseTool
from pydantic import Field

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chromadb")
COLLECTION_NAME = "news_articles"

# ── Shared embedding model (loaded once) ─────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Returns (or creates) the persistent ChromaDB vector store."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ── Tool 1: Web Search + ChromaDB Ingestion ───────────────────────────────────

class WebSearchTool(BaseTool):
    """
    CrewAI tool for Agent 1 (News Research Specialist).
    Searches the web via Tavily and ingests results into ChromaDB.
    """
    name: str = "WebSearchTool"
    description: str = (
        "Searches the web for the latest news on a given topic using Tavily, "
        "then stores the retrieved articles in ChromaDB for later RAG retrieval. "
        "Input should be a search query string."
    )

    def _run(self, query: str) -> str:
        try:
            from tavily import TavilyClient
        except ImportError:
            return "ERROR: tavily-python not installed. Run: pip install tavily-python"

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return (
                "ERROR: TAVILY_API_KEY not found in environment. "
                "Sign up free at https://tavily.com and add the key to .env"
            )

        # ── Step 1: Search ────────────────────────────────────────────────────
        try:
            client = TavilyClient(api_key=api_key)
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=8,
                include_raw_content=True,
            )
        except Exception as e:
            return f"Tavily search error: {e}"

        if not results.get("results"):
            return f"No results found for query: {query}"

        # ── Step 2: Prepare documents ─────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
        )

        docs = []
        for r in results["results"]:
            content = r.get("raw_content") or r.get("content", "")
            if not content:
                continue

            chunks = splitter.create_documents(
                texts=[content],
                metadatas=[{
                    "source": r.get("url", ""),
                    "title": r.get("title", ""),
                    "query": query,
                    "doc_id": str(uuid.uuid4()),
                }],
            )
            docs.extend(chunks)

        if not docs:
            return "Search returned results but no extractable content."

        # ── Step 3: Ingest into ChromaDB ──────────────────────────────────────
        try:
            vs = get_vectorstore()
            vs.add_documents(docs)
            vs.persist()
        except Exception as e:
            return f"ChromaDB ingestion error: {e}"

        sources = list({r.get("title", r.get("url", "")) for r in results["results"]})
        return (
            f"✅ Successfully ingested {len(docs)} document chunks from "
            f"{len(results['results'])} sources into ChromaDB.\n"
            f"Sources: {', '.join(sources[:5])}"
        )


# ── Tool 2: RAG Retrieval + Gemini Generation ─────────────────────────────────

class RAGRetrieveTool(BaseTool):
    """
    CrewAI tool for Agent 2 (News Analyst & Summarizer).
    Retrieves relevant chunks from ChromaDB and generates a structured
    bullet-point summary using Gemini.
    """
    name: str = "RAGRetrieveTool"
    description: str = (
        "Retrieves the most relevant news document chunks from ChromaDB using "
        "semantic similarity search, then uses Gemini AI to generate a concise, "
        "structured bullet-point news summary. "
        "Input should be the topic or query to summarize."
    )

    def _run(self, query: str) -> str:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                "ERROR: GEMINI_API_KEY not found. "
                "Get a free key at https://aistudio.google.com"
            )

        # ── Step 1: Retrieve from ChromaDB ────────────────────────────────────
        try:
            vs = get_vectorstore()
            retriever = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},
            )
            relevant_docs = retriever.get_relevant_documents(query)
        except Exception as e:
            return f"ChromaDB retrieval error: {e}"

        if not relevant_docs:
            return (
                "No documents found in ChromaDB for this query. "
                "Ensure the News Research Specialist has run first."
            )

        # ── Step 2: Build context ─────────────────────────────────────────────
        context_parts = []
        seen_sources = set()
        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "")
            if source not in seen_sources:
                seen_sources.add(source)
                context_parts.append(
                    f"[Source: {title or source}]\n{doc.page_content}"
                )

        context = "\n\n---\n\n".join(context_parts)

        # ── Step 3: Generate with Gemini ──────────────────────────────────────
        prompt = f"""You are a world-class news analyst. Based ONLY on the provided news sources below, 
create a comprehensive, well-structured news summary about: "{query}"

NEWS SOURCES:
{context}

FORMAT YOUR RESPONSE AS FOLLOWS:

## 📰 News Summary: {query}

### 🔑 Key Highlights
- [Most important point 1]
- [Most important point 2]  
- [Most important point 3]
- [Most important point 4]
- [Most important point 5]

### 📊 Detailed Analysis
[2-3 paragraphs with deeper context and analysis]

### 🌍 Global Impact
- [Impact point 1]
- [Impact point 2]
- [Impact point 3]

### 🔮 What to Watch
- [Forward-looking point 1]
- [Forward-looking point 2]

### 📌 Sources Referenced
{chr(10).join(f'- {s}' for s in list(seen_sources)[:8])}

Be factual, balanced, and base everything strictly on the provided sources."""

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as e:
            return f"Gemini generation error: {e}"
