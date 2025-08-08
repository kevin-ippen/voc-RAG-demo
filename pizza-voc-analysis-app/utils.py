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
    """
    Wrapper that tolerates multiple VectorSearchClient versions.
    If the import/constructor fails, uses REST fallback through WorkspaceClient.api_client.
    """

    def __init__(self, ws: WorkspaceClient):
        self.ws = ws
        self.endpoint_name = (
            os.getenv("VECTOR_SEARCH_ENDPOINT")
            or os.getenv("VECTORSEARCH_ENDPOINT")
        )
        self.index_fqn = (
            os.getenv("VECTOR_INDEX_NAME")
            or os.getenv("VECTORSEARCH_INDEX")
        )

        if not self.endpoint_name:
            raise ValueError("VECTOR_SEARCH_ENDPOINT is not set.")
        if not self.index_fqn or self.index_fqn.count(".") != 2:
            raise ValueError(
                "VECTOR_INDEX_NAME must be fully qualified as catalog.schema.index "
                "(e.g., main.vs_schema.voc_chunks_index)."
            )

        self._client = self._build_client_or_none()

    def _build_client_or_none(self):
        """Try many signatures; return client or None (REST fallback)."""
        if VectorSearchClient is None:
            return None

        attempts = [
            {"workspace_client": self.ws},                 # newer
            {"workspace": self.ws},                        # some mid versions logged 'workspace' (rare)
            {"api_client": self.ws.api_client},            # older accepted ApiClient
            {"config": self.ws.config},                    # some accepted Config directly
            {"workspace_url": self.ws.config.host, "personal_access_token": self.ws.config.token},  # very old
        ]

        last_err = None
        for kwargs in attempts:
            try:
                return VectorSearchClient(**kwargs)
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        # Could not construct a typed client — we'll REST-fallback in search()
        # Keep the last error for debug
        self._ctor_error = last_err
        return None

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        **filters: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not query_text:
            return []

        meta_filters = {k: v for k, v in filters.items() if v}

        if self._client is not None:
            # Use the typed client, handling top_k/num_results rename
            kwargs = {
                "index_name": self.index_fqn,
                "endpoint_name": self.endpoint_name,
                "query_text": query_text,
            }
            try:
                res = self._client.indexes.query_index(num_results=top_k, filters=meta_filters, **kwargs)
            except TypeError:
                res = self._client.indexes.query_index(top_k=top_k, filters=meta_filters, **kwargs)

            hits = getattr(res, "result", None) or getattr(res, "data", None) or []
            return self._normalize_hits(hits)

        # ---------- REST fallback ----------
        # Endpoint (stable across versions):
        # POST /api/2.0/vector-search/indexes/{index_fqn}/query
        body: Dict[str, Any] = {
            "query": {"query_text": query_text},
            "num_results": top_k,
            "endpoint_name": self.endpoint_name,
        }
        if meta_filters:
            body["filters"] = {"metadata": meta_filters}

        try:
            raw = self.ws.api_client.do(
                "POST",
                f"/api/2.0/vector-search/indexes/{self.index_fqn}/query",
                body=body,
            )
        except Exception as e:
            detail = getattr(self, "_ctor_error", None)
            raise RuntimeError(
                "Vector search failed. Client ctor error: "
                f"{detail}; REST error: {e}"
            )

        # Older responses usually return {"data": [ ... ]} or {"result": [...]}
        hits = raw.get("result") or raw.get("data") or []
        return self._normalize_hits(hits)

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
