# config.py
import os
from dataclasses import dataclass, field
from typing import List, Dict

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default

@dataclass
class Config:
    # Core DS/LLM endpoints (env-first)
    VECTOR_SEARCH_ENDPOINT: str = _env("VECTOR_SEARCH_ENDPOINT", "dbdemos_vs_endpoint")
    VECTOR_INDEX_NAME: str = _env("VECTOR_INDEX_NAME", "users.kevin_ippen.voc_chunks_index")
    LLM_ENDPOINT_NAME: str = _env("LLM_ENDPOINT", "databricks-gpt-oss-20b")

    # Query tuning
    DEFAULT_NUM_RESULTS: int = 5
    MAX_NUM_RESULTS: int = 50  # give yourself more headroom

    # LLM params
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 500

    # Filters — align to actual VS column values (edit if yours differ)
    SATISFACTION_LEVELS: List[str] = field(default_factory=lambda: ["Satisfied", "Neutral", "Dissatisfied"])
    SERVICE_METHODS: List[str] = field(default_factory=lambda: ["Delivery", "Pickup"])
    CUSTOMER_TYPES: List[str] = field(default_factory=lambda: ["New", "Returning"])
    ORDER_SOURCES: List[str] = field(default_factory=lambda: ["App", "Web", "Phone"])

    # Columns to fetch from the index (match your schema)
    SEARCH_COLUMNS: List[str] = field(default_factory=lambda: [
        "id",
        "text",
        "satisfaction",
        "service_method",
        "customer_type",
        "order_source",
        "order_date",
        "service_time",
    ])

    # UI
    APP_TITLE: str = "🍕 Pizza Company VOC Analysis"
    APP_DESCRIPTION: str = "AI-Powered Customer Feedback Insights using RAG and Vector Search"

    SAMPLE_QUESTIONS: List[str] = field(default_factory=lambda: [
        "What do customers complain about most?",
        "How satisfied are customers with delivery?",
        "What do customers love about our pizza?",
        "Are there issues with order accuracy?",
        "What makes customers dissatisfied lately?",
    ])

    QUICK_ANALYSIS_TOPICS: List[str] = field(default_factory=lambda: [
        "pizza quality", "delivery service", "customer service",
        "order accuracy", "wait times", "mobile app", "website ordering"
    ])

    RAG_PROMPT_TEMPLATE: str = (
        "You are a helpful assistant analyzing customer feedback for a pizza company.\n"
        "Based on the customer comments provided below, answer the question accurately and provide insights.\n\n"
        "Customer Feedback Context:\n{context}\n\nQuestion: {question}\n\n"
        "Instructions:\n"
        "- Answer based only on the provided customer feedback\n"
        "- Provide specific examples from the comments when relevant\n"
        "- If the context doesn't contain enough information, say so\n"
        "- Identify patterns and trends in customer sentiment when applicable\n"
        "- Be concise but informative\n"
        "- Focus on actionable insights for business improvement\n\n"
        "Answer:"
    )

    CHART_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "Satisfied": "#2E8B57",
        "Neutral": "#FFD700",
        "Dissatisfied": "#FF6347",
    })

    ERROR_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        "vector_search_init": "Failed to initialize vector search. Please check your configuration.",
        "search_failed": "Search failed. Please try a different query.",
        "no_results": "No matching customer comments found. Try different search terms.",
        "rag_failed": "Failed to generate insights. Please try again.",
        "llm_failed": "Failed to generate AI response. Please check model serving endpoint."
    })

    @classmethod
    def validate_config(cls) -> None:
        problems = []
        vsi = _env("VECTOR_INDEX_NAME", cls.VECTOR_INDEX_NAME)
        vse = _env("VECTOR_SEARCH_ENDPOINT", cls.VECTOR_SEARCH_ENDPOINT)
        host = _env("DATABRICKS_HOST")

        if not vse:
            problems.append("VECTOR_SEARCH_ENDPOINT is missing.")
        if not vsi or "." not in vsi:
            problems.append("VECTOR_INDEX_NAME is missing or not a full path (catalog.schema.name).")
        if not host or not host.startswith("http"):
            problems.append("DATABRICKS_HOST is missing or invalid (must include https://).")
        if not _env("DATABRICKS_AUTH_TYPE") and not _env("DATABRICKS_TOKEN"):
            # SP injected creds are fine even if AUTH_TYPE unset, but warn if all auth hints are absent
            cid, csec = _env("DATABRICKS_CLIENT_ID"), _env("DATABRICKS_CLIENT_SECRET")
            if not (cid and csec):
                problems.append("No clear auth configured (set DATABRICKS_AUTH_TYPE=oauth-m2m or provide a PAT).")

        if problems:
            raise ValueError("Config validation failed: " + "; ".join(problems))
