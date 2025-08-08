import os
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config, ApiClient

# Try to import the VS client (module paths have shifted over time)
VS_CLIENT_IMPORT_ERRORS = []
VectorSearchClient = None
for mod in (
    "databricks.vector_search.client",
    "databricks.vector_search",  # older
):
    try:
        m = __import__(mod, fromlist=["VectorSearchClient"])
        VectorSearchClient = getattr(m, "VectorSearchClient")
        break
    except Exception as e:
        VS_CLIENT_IMPORT_ERRORS.append((mod, str(e)))
        continue


def _make_ws() -> WorkspaceClient:
    """Create a WorkspaceClient using SP (oauth-m2m) or PAT, falling back to default env."""
    host = os.getenv("DATABRICKS_HOST")
    cid = os.getenv("DATABRICKS_CLIENT_ID")
    csec = os.getenv("DATABRICKS_CLIENT_SECRET")
    auth_type = (os.getenv("DATABRICKS_AUTH_TYPE") or "").lower()
    pat = os.getenv("DATABRICKS_TOKEN")

    # Prefer SP if supplied (and/or oauth-m2m signaled)
    if host and cid and csec and (auth_type in ("", "oauth-m2m", "m2m")):
        return WorkspaceClient(
            config=Config(host=host, client_id=cid, client_secret=csec, auth_type="oauth-m2m")
        )

    # PAT
    if host and pat:
        return WorkspaceClient(config=Config(host=host, token=pat))

    # SDK default (env/metadata)
    try:
        return WorkspaceClient()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Databricks auth. Set either SP (HOST, CLIENT_ID, CLIENT_SECRET, AUTH_TYPE=oauth-m2m) "
            "or PAT (HOST, TOKEN). Details: " + str(e)
        )


class VectorSearchManager:
    def __init__(self, ws: WorkspaceClient):
        self.ws = ws
        self.endpoint_name = os.getenv("VECTOR_SEARCH_ENDPOINT") or os.getenv("VECTORSEARCH_ENDPOINT")
        self.index_fqn = os.getenv("VECTOR_INDEX_NAME") or os.getenv("VECTORSEARCH_INDEX")
        if not self.endpoint_name:
            raise ValueError("VECTOR_SEARCH_ENDPOINT is not set.")
        if not self.index_fqn or self.index_fqn.count(".") != 2:
            raise ValueError("VECTOR_INDEX_NAME must be fully qualified as catalog.schema.index")

        self._client = self._build_client_or_none()
        self._available_columns = self._discover_columns()
        self._text_col = self._pick_text_col(self._available_columns)  # <- choose the right text column

    def _build_client_or_none(self):
        if VectorSearchClient is None:
            return None
        attempts = [
            {"workspace_client": self.ws},
            {"workspace": self.ws},
            {"api_client": self.ws.api_client},
            {"config": self.ws.config},
            {"workspace_url": self.ws.config.host, "personal_access_token": self.ws.config.token},
        ]
        last_err = None
        for kwargs in attempts:
            try:
                return VectorSearchClient(**kwargs)
            except Exception as e:
                last_err = e
                continue
        self._ctor_error = last_err
        return None

    def _discover_columns(self) -> set:
        try:
            desc = self.ws.api_client.do("GET", f"/api/2.0/vector-search/indexes/{self.index_fqn}")
            cols = desc.get("columns") or []
            names = set()
            for c in cols:
                if isinstance(c, str):
                    names.add(c)
                elif isinstance(c, dict) and "name" in c:
                    names.add(c["name"])
            return names or {"text"}  # fallback
        except Exception:
            return {"text"}

    @staticmethod
    def _pick_text_col(names: set) -> str:
        # order of preference for common text fields
        prefs = ["text", "chunk", "content", "body", "page_text", "document_text", "value"]
        for p in prefs:
            if p in names:
                return p
        # heuristic: first column containing 'text' or 'content'
        for n in names:
            ln = n.lower()
            if "text" in ln or "content" in ln or "chunk" in ln:
                return n
        # last resort: arbitrary first name
        return next(iter(names))

    def search(self, query_text: str, top_k: int = 10, **filters: Optional[str]) -> List[Dict[str, Any]]:
        if not query_text:
            return []

        # TEMP: ignore filters for the sanity pass; we’ll re-enable after we see results
        meta_filters = {k: v for k, v in filters.items() if v}

    @staticmethod
    def _normalize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for h in hits:
            # try common shapes
            text = h.get("text") or h.get("chunk") or (h.get("document") or {}).get("text")
            score = h.get("score") or h.get("similarity")
            meta = h.get("metadata") or (h.get("document") or {}).get("metadata") or {}
            row = {"text": text, "score": score}
            if isinstance(meta, dict):
                row.update(meta)
            out.append(row)
        return out
