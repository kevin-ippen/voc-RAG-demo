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

    def _extract_hits(self, raw: dict) -> list:
        """
        Handle older VS response shapes:
        - {"columns": ["text", ...], "data": [ ["a", ...], ["b", ...] ]}
        - {"data": [{"text": "...", ...}, ...]}
        - {"result": [{"text": "...", ...}, ...]}
        - ["a", "b", ...]  (plain strings)
        """
        if not isinstance(raw, dict):
            # Some very old endpoints return just a list
            return raw if isinstance(raw, list) else []

        # Columnar (columns + data)
        cols = raw.get("columns")
        data = raw.get("data")
        if isinstance(cols, list) and isinstance(data, list):
            out = []
            for row in data:
                if isinstance(row, list):
                    out.append({cols[i]: row[i] for i in range(min(len(cols), len(row)))})
                elif isinstance(row, dict):
                    out.append(row)
                elif isinstance(row, str):
                    out.append({"text": row})
            return out

        # Dict lists under known keys
        for key in ("result", "data", "hits", "items"):
            val = raw.get(key)
            if isinstance(val, list):
                return val

        # Plain list of strings?
        if isinstance(raw, list):
            return raw

        return []

    def _normalize_hits(self, hits: list) -> list[dict]:
        """
        Accepts strings, dicts, or mixed lists and produces a list of dicts with at least {'text', 'score?'}.
        """
        out = []
        for h in hits:
            if isinstance(h, str):
                out.append({"text": h, "score": None})
                continue
            if not isinstance(h, dict):
                # Best effort
                out.append({"text": str(h), "score": None})
                continue

            # Common shapes
            text = (
                h.get("text")
                or h.get("chunk")
                or (h.get("document") or {}).get("text")
                or h.get("value")
            )
            score = h.get("score") or h.get("similarity") or h.get("distance")
            meta = (
                h.get("metadata")
                or (h.get("document") or {}).get("metadata")
                or {}
            )

            row = {"text": text, "score": score}
            if isinstance(meta, dict):
                row.update(meta)

            # Copy a few likely fields if present
            for k in ("id", "source", "document_id"):
                if k in h and k not in row:
                    row[k] = h[k]

            # If text is still missing but there’s a single key, pick it
            if row.get("text") is None and len(h) == 1:
                only_key = next(iter(h))
                row["text"] = h[only_key]

            out.append(row)
        return out

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
            # Use typed client, handling num_results/top_k rename
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

        # ---------- REST fallback for older backends ----------
        # Older API expects top-level 'query_text' or 'query_vector', and a 'columns' array.
        wanted = ["text", "metadata", "id", "source", "document_id"]
        request_cols = [c for c in wanted if c in getattr(self, "_available_columns", {"text"})]
        if not request_cols:
            request_cols = ["text"]

        body: Dict[str, Any] = {
            "query_text": query_text,       # top-level for old API
            "num_results": top_k,
            "endpoint_name": self.endpoint_name,
            "columns": request_cols,        # ask only for columns that exist
            "return_scores": True,
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
            raise RuntimeError(f"Vector search failed (REST): {e}")

        hits = self._extract_hits(raw)
        return self._normalize_hits(hits)

    def describe_index(self) -> Dict[str, Any]:
        """Fetch index metadata to see available columns/fields."""
        return self.ws.api_client.do(
            "GET", f"/api/2.0/vector-search/indexes/{self.index_fqn}"
        )


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
