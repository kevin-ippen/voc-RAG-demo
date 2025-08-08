import os
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# Vector Search SDK compat (older vs newer signatures)
try:
    from databricks.vector_search.client import VectorSearchClient
except Exception:
    # very old name (unlikely now, but safe)
    from databricks.vector_search import VectorSearchClient  # type: ignore


# -------------------- Auth / WS --------------------
def _make_ws() -> WorkspaceClient:
    """
    Create a WorkspaceClient using one of:
      1) Service Principal (CLIENT_ID/SECRET + HOST)
      2) PAT (TOKEN + HOST)
      3) Default env/metadata if available
    Raises with a clear message if nothing is usable.
    """
    host = os.getenv("DATABRICKS_HOST")

    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    auth_type = (os.getenv("DATABRICKS_AUTH_TYPE") or "").lower()

    pat = os.getenv("DATABRICKS_TOKEN")

    # Prefer explicit service principal when provided (or when oauth-m2m is set)
    if client_id and client_secret and host and (auth_type in ("", "oauth-m2m", "m2m")):
        cfg = Config(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )
        return WorkspaceClient(config=cfg)

    # PAT fallback
    if pat and host:
        cfg = Config(host=host, token=pat)
        return WorkspaceClient(config=cfg)

    # Final fallback to whatever env/metadata the SDK can pick up (e.g., dev boxes)
    try:
        return WorkspaceClient()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Databricks auth. Set **either**:\n"
            " - Service principal: DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, "
            "  (optional) DATABRICKS_AUTH_TYPE=oauth-m2m\n"
            " - OR a PAT: DATABRICKS_HOST, DATABRICKS_TOKEN\n"
            f"Details: {e}"
        )


# -------------------- Vector Search Manager --------------------
class VectorSearchManager:
    """
    Thin wrapper around Vector Search queries that:
      - Handles SDK signature differences
      - Centralizes endpoint + index configuration
      - Normalizes results
    """

    def __init__(self, ws: WorkspaceClient):
        self.ws = ws

        # Allow env override; defer to config.py if you have one, else pure env is fine
        self.endpoint_name = os.getenv("VECTOR_SEARCH_ENDPOINT") or os.getenv("VECTORSEARCH_ENDPOINT")
        self.index_fqn = os.getenv("VECTOR_INDEX_NAME") or os.getenv("VECTORSEARCH_INDEX")

        if not self.endpoint_name:
            raise ValueError("VECTOR_SEARCH_ENDPOINT is not set.")
        if not self.index_fqn or self.index_fqn.count(".") != 2:
            raise ValueError(
                "VECTOR_INDEX_NAME must be fully qualified as catalog.schema.index "
                "(e.g., main.vs_schema.voc_chunks_index)."
            )

        # Build client with best-effort compatibility
        self.vs = None
        last_err = None
        for kw in ("workspace_client", "workspace"):
            try:
                self.vs = VectorSearchClient(**{kw: self.ws})
                break
            except TypeError as e:
                last_err = e
                continue
        if self.vs is None:
            raise RuntimeError(
                f"Failed to create VectorSearchClient; upgrade 'databricks-vectorsearch' package. Last error: {last_err}"
            )

    # ---- Public API ----
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        **filters: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search. Optional filters are passed as metadata filters if your index supports them.
        Returns a list of dicts with keys like: text, score, and metadata fields (if present).
        """
        if not query_text:
            return []

        # Build filter dict excluding falsy values
        meta_filters = {k: v for k, v in filters.items() if v}

        # Call VS query with param-name compatibility
        kwargs = {
            "index_name": self.index_fqn,
            "endpoint_name": self.endpoint_name,
            "query_text": query_text,
        }

        # Handle num_results/top_k variation across SDK versions
        try:
            result = self.vs.indexes.query_index(num_results=top_k, filters=meta_filters, **kwargs)
        except TypeError:
            result = self.vs.indexes.query_index(top_k=top_k, filters=meta_filters, **kwargs)

        # Normalize results
        out: List[Dict[str, Any]] = []
        hits = getattr(result, "result", None) or getattr(result, "data", None) or []
        for h in hits:
            # Try common shapes
            text = h.get("text") or h.get("chunk") or h.get("document", {}).get("text")
            score = h.get("score") or h.get("similarity")
            meta = h.get("metadata") or h.get("document", {}).get("metadata") or {}
            row = {"text": text, "score": score}
            row.update(meta if isinstance(meta, dict) else {})
            out.append(row)
        return out
