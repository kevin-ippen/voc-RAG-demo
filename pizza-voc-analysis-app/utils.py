# utils.py
"""
Utility classes and functions for the Pizza VOC Analysis Databricks App.
Hardened for auth (SP/PAT/OBO), env-driven config, and schema-safe parsing.
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config as SDKConfig
from databricks.vector_search.client import VectorSearchClient

try:
    # Optional – if you keep your prompts/labels in config.py
    from config import Config as AppConfig
except Exception:
    AppConfig = None  # it's fine if you don't have one

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------- Auth helpers ---------------------- #
def _make_ws(obo_token: Optional[str] = None) -> WorkspaceClient:
    """
    Build a WorkspaceClient with the strongest available auth:
      1) Injected Service Principal (oauth-m2m) if CLIENT_ID/SECRET present
      2) PAT via DATABRICKS_TOKEN
      3) OBO bearer token passed in explicitly (viewer identity)
    """
    host = os.getenv("DATABRICKS_HOST")
    if not host or not host.startswith("http"):
        raise RuntimeError("DATABRICKS_HOST missing or invalid (must include https://)")

    cid = os.getenv("DATABRICKS_CLIENT_ID")
    csec = os.getenv("DATABRICKS_CLIENT_SECRET")
    auth_type = os.getenv("DATABRICKS_AUTH_TYPE")
    token = os.getenv("DATABRICKS_TOKEN")
    token_endpoint = os.getenv("DATABRICKS_OAUTH_TOKEN_ENDPOINT")  # usually not needed on AWS

    # 1) Service Principal (preferred)
    if cid and csec:
        cfg = SDKConfig(
            host=host,
            auth_type="oauth-m2m",  # force SP path
            client_id=cid,
            client_secret=csec,
            oauth_token_endpoint=token_endpoint,  # harmless if None on AWS
        )
        return WorkspaceClient(config=cfg)

    # 2) PAT
    if token:
        cfg = SDKConfig(host=host, token=token)
        return WorkspaceClient(config=cfg)

    # 3) OBO (viewer token passed by caller)
    if obo_token:
        cfg = SDKConfig(host=host, token=obo_token)
        return WorkspaceClient(config=cfg)

    # If user explicitly set oauth-m2m but the creds are missing, tell them
    if auth_type == "oauth-m2m":
        raise RuntimeError(
            "DATABRICKS_AUTH_TYPE=oauth-m2m but CLIENT_ID/SECRET not found. "
            "Ensure the app runs as a service principal or set the env vars."
        )

    raise RuntimeError(
        "No credentials found. Provide Service Principal (CLIENT_ID/SECRET), a PAT (DATABRICKS_TOKEN), "
        "or pass an OBO token to VectorSearchManager(..., obo_token=...)."
    )


# ---------------------- Config helpers ---------------------- #
def _env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else fallback


def _resolve_config() -> Tuple[str, str, List[str]]:
    """
    Resolve endpoint, index, and columns from env first, then AppConfig defaults.
    """
    endpoint = _env("VECTOR_SEARCH_ENDPOINT", getattr(AppConfig, "VECTOR_SEARCH_ENDPOINT", None) if AppConfig else None)
    index = _env("VECTOR_INDEX_NAME", getattr(AppConfig, "VECTOR_INDEX_NAME", None) if AppConfig else None)

    default_cols = [
        "id",
        "text",
        "satisfaction",
        "service_method",
        "customer_type",
        "order_source",
        "order_date",
        "service_time",
    ]
    columns = getattr(AppConfig, "SEARCH_COLUMNS", default_cols) if AppConfig else default_cols

    if not endpoint:
        raise ValueError("VECTOR_SEARCH_ENDPOINT is not set (env or config).")
    if not index or "." not in index:
        raise ValueError("VECTOR_INDEX_NAME is missing or not a full path (catalog.schema.name).")

    return endpoint, index, columns


# ---------------------- Vector Search ---------------------- #
class VectorSearchManager:
    """Manages vector search operations for customer feedback."""

    def __init__(
        self,
        ws: Optional[WorkspaceClient] = None,
        endpoint_name: Optional[str] = None,
        index_name: Optional[str] = None,
        search_columns: Optional[List[str]] = None,
        obo_token: Optional[str] = None,
    ):
        """
        Initialize the vector search manager.

        Args:
            ws: Optional prebuilt WorkspaceClient (auth already configured)
            endpoint_name: VS endpoint name (env/config used if None)
            index_name: Fully-qualified index name (catalog.schema.index)
            search_columns: List of columns to retrieve from the index
            obo_token: Optional viewer token to use OBO auth (if SP/PAT not configured)
        """
        if ws is None:
            ws = _make_ws(obo_token=obo_token)

        self.client = VectorSearchClient(workspace=ws, disable_notice=True)

        # Resolve endpoint/index/columns
        env_endpoint, env_index, env_cols = _resolve_config()
        self.endpoint_name = endpoint_name or env_endpoint
        self.index_name = index_name or env_index
        self.search_columns = search_columns or env_cols

        # Basic validation
        if not self.endpoint_name:
            raise ValueError("Vector Search endpoint name is required.")
        if not self.index_name or "." not in self.index_name:
            raise ValueError("Vector Search index name must be a full path (catalog.schema.index).")

        # Lazily created index handle
        self._index = None

    def _get_index(self):
        """Get and cache the vector search index handle."""
        if self._index is None:
            self._index = self.client.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name,
            )
        return self._index

    @staticmethod
    def _build_expr(
        satisfaction: Optional[str] = None,
        service_method: Optional[str] = None,
        customer_type: Optional[str] = None,
        order_source: Optional[str] = None,
    ) -> Optional[str]:
        """Build a VS expr filter string."""
        parts = []
        if satisfaction:
            parts.append(f"satisfaction = '{satisfaction}'")
        if service_method:
            parts.append(f"service_method = '{service_method}'")
        if customer_type:
            parts.append(f"customer_type = '{customer_type}'")
        if order_source:
            parts.append(f"order_source = '{order_source}'")
        return " AND ".join(parts) if parts else None

    def search_feedback(
        self,
        query_text: str,
        num_results: int = 25,
        satisfaction: Optional[str] = None,
        service_method: Optional[str] = None,
        customer_type: Optional[str] = None,
        order_source: Optional[str] = None,
        extra_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search with optional filters. Returns list of dict rows using server-returned column names.
        """
        idx = self._get_index()

        # Compose expr filter
        base_expr = self._build_expr(satisfaction, service_method, customer_type, order_source)
        expr = f"({base_expr}) AND ({extra_expr})" if (base_expr and extra_expr) else (extra_expr or base_expr)
        filters = {"expr": expr} if expr else None

        params = {
            "query_text": query_text,
            "columns": self.search_columns,
            "num_results": max(1, int(num_results)),
        }
        if filters:
            params["filters"] = filters

        try:
            res = idx.similarity_search(**params) or {}
            result = res.get("result", {})
            rows = result.get("data_array", []) or []
            cols = result.get("columns", self.search_columns)

            # Zip into dict rows (schema-safe)
            out = [dict(zip(cols, r)) for r in rows]

            # If a score column exists, keep it; otherwise drop silently
            return out
        except Exception as e:
            log.error("Vector search failed: %s", e, exc_info=True)
            return []

    # Convenience: older API that returns lists if your UI expects it
    def search_feedback_as_lists(
        self,
        *args,
        **kwargs,
    ) -> List[List[Any]]:
        dict_rows = self.search_feedback(*args, **kwargs)
        if not dict_rows:
            return []
        cols = list(dict_rows[0].keys())
        return [[row.get(c) for c in cols] for row in dict_rows]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Quick sample-based stats (best-effort)."""
        sample = self.search_feedback("pizza", num_results=100)
        if not sample:
            return {"error": "No data available"}
        def dist(key: str) -> Dict[str, int]:
            vals = [r.get(key, "Unknown") for r in sample]
            return {v: vals.count(v) for v in set(vals)}
        return {
            "total_sample_size": len(sample),
            "satisfaction_distribution": dist("satisfaction"),
            "service_method_distribution": dist("service_method"),
        }


# ---------------------- RAG helpers (optional) ---------------------- #
class RAGPipeline:
    """Construct prompts and package retrieval results for your LLM caller."""

    def __init__(self, vsm: VectorSearchManager, prompt_template: Optional[str] = None):
        self.vsm = vsm
        if prompt_template:
            self.template = prompt_template
        elif AppConfig and hasattr(AppConfig, "RAG_PROMPT_TEMPLATE"):
            self.template = AppConfig.RAG_PROMPT_TEMPLATE
        else:
            self.template = (
                "You are a helpful assistant analyzing customer feedback.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Instructions:\n- Answer only from the context.\n- Be concise and actionable.\n\nAnswer:"
            )

    @staticmethod
    def _join_contexts(contexts: List[str]) -> str:
        return "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        return self.template.format(context=self._join_contexts(contexts), question=question)

    def ask(
        self,
        question: str,
        num_contexts: int = 5,
        satisfaction: Optional[str] = None,
        service_method: Optional[str] = None,
        customer_type: Optional[str] = None,
        order_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows = self.vsm.search_feedback(
            query_text=question,
            num_results=num_contexts,
            satisfaction=satisfaction,
            service_method=service_method,
            customer_type=customer_type,
            order_source=order_source,
        )
        contexts = [r.get("text", "") for r in rows]
        meta = [
            {
                "id": r.get("id"),
                "satisfaction": r.get("satisfaction"),
                "service_method": r.get("service_method"),
                "score": r.get("score", r.get("_score", None)),
            }
            for r in rows
        ]
        return {
            "question": question,
            "contexts_found": len(contexts),
            "contexts": contexts,
            "metadata": meta,
            "rag_prompt": self.build_prompt(question, contexts),
            "status": "success",
        }
