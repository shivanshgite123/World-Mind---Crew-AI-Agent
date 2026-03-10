"""
CrewAI Crew Definition

Two agents in sequential process:
  1. News Research Specialist  → searches web + stores in ChromaDB
  2. News Analyst & Summarizer → retrieves from ChromaDB + generates summary
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from gemini_llm import get_gemini_llm
from rag_chain import WebSearchTool, RAGRetrieveTool

load_dotenv()


def run_news_crew(query: str) -> dict:
    """
    Orchestrates the full RAG pipeline via CrewAI for a given news query.

    Args:
        query: The news topic to search and summarize

    Returns:
        dict with keys: summary (str), status (str), error (str|None)
    """
    try:
        # ── LLM ──────────────────────────────────────────────────────────────
        llm = get_gemini_llm(temperature=0.2)

        # ── Tools ─────────────────────────────────────────────────────────────
        web_search_tool = WebSearchTool()
        rag_retrieve_tool = RAGRetrieveTool()

        # ── Agent 1: News Research Specialist ─────────────────────────────────
        research_agent = Agent(
            role="News Research Specialist",
            goal=(
                "Search the internet for the latest, most relevant news articles "
                "about the given topic and store them in the vector database "
                "for downstream analysis."
            ),
            backstory=(
                "You are an elite news researcher with 20 years of experience "
                "at major global news organizations. You have an exceptional ability "
                "to find authoritative, credible, and up-to-date news from across "
                "the globe. You are methodical, thorough, and always verify sources. "
                "You understand how to craft precise search queries to maximize "
                "the quality and coverage of results."
            ),
            tools=[web_search_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

        # ── Agent 2: News Analyst & Summarizer ────────────────────────────────
        analyst_agent = Agent(
            role="News Analyst & Summarizer",
            goal=(
                "Retrieve the stored news articles from the vector database, "
                "analyze them for key themes, facts, and global significance, "
                "then produce a comprehensive, well-structured summary."
            ),
            backstory=(
                "You are a senior news analyst and editor who has worked for "
                "Reuters, BBC, and The Economist. You excel at synthesizing "
                "complex, multi-source information into clear, balanced, and "
                "insightful summaries. Your writing is precise, neutral, and "
                "deeply informative. You always cite your sources and provide "
                "context for global readers."
            ),
            tools=[rag_retrieve_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

        # ── Task 1: Research ───────────────────────────────────────────────────
        research_task = Task(
            description=(
                f"Search the web for the latest news about: '{query}'\n\n"
                "Steps:\n"
                "1. Use the WebSearchTool with the query to find recent news articles\n"
                "2. Ensure articles are stored in ChromaDB\n"
                "3. Report back on what was found and stored\n\n"
                "Be thorough — search for at least 5-8 distinct news sources."
            ),
            expected_output=(
                "A confirmation message listing the number of articles found, "
                "the number of document chunks stored in ChromaDB, and a brief "
                "list of source names/titles that were retrieved."
            ),
            agent=research_agent,
        )

        # ── Task 2: Analyze & Summarize ────────────────────────────────────────
        analysis_task = Task(
            description=(
                f"Retrieve all stored news about '{query}' from ChromaDB and "
                "create a comprehensive, structured summary.\n\n"
                "Steps:\n"
                "1. Use the RAGRetrieveTool with the query to retrieve relevant articles\n"
                "2. Analyze the retrieved content for key facts, themes, and significance\n"
                "3. Produce a well-formatted summary with bullet points and sections\n\n"
                "The summary must be accurate, neutral, and well-organized."
            ),
            expected_output=(
                "A comprehensive news summary in Markdown format with:\n"
                "- Key highlights as bullet points\n"
                "- Detailed analysis paragraphs\n"
                "- Global impact assessment\n"
                "- What to watch going forward\n"
                "- List of sources referenced"
            ),
            agent=analyst_agent,
            context=[research_task],
        )

        # ── Crew ───────────────────────────────────────────────────────────────
        crew = Crew(
            agents=[research_agent, analyst_agent],
            tasks=[research_task, analysis_task],
            process=Process.sequential,
            verbose=True,
        )

        # ── Kickoff ────────────────────────────────────────────────────────────
        result = crew.kickoff(inputs={"query": query})

        # Handle both string and CrewOutput return types
        summary_text = str(result) if not isinstance(result, str) else result

        return {
            "status": "success",
            "summary": summary_text,
            "error": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "",
            "error": str(e),
        }
