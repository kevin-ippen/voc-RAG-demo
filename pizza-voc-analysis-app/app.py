# app.py — Pizza VOC Explorer (clean, self-contained)

from __future__ import annotations
import os
import json
import logging
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd

# Databricks SDK / clients
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config as SDKConfig
from databricks.vector_search.client import VectorSearchClient
from databricks import sql
from mlflow.deployments import get_deploy_client

# ------------- Streamlit setup -------------
st.set_page_config(page_title="Pizza VOC Explorer", page_icon="🍕", layout="wide")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pizza-voc-app")

# ------------- Environment / Config -------------
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # MUST include https://
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT")         # e.g., dbdemos_vs_endpoint
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")                   # e.g., users.kevin.voc_index
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")                             # e.g., databricks-gpt-oss-20b

SEARCH_COLUMNS: List[str] = [
    # match your index schema; safe defaults (adjust as needed)
    "id",
    "text",
    "satisfaction",
    "service_method",
    "customer_type",
    "order_source",
    "order_date",
    "service_time",
]

SATISFACTION_VALUES = ["", "Satisfied", "Neutral", "Dissatisfied"]  # <- align to your real values
SERVICE_METHOD_VALUES = ["", "Delivery", "Pickup"]
CUSTOMER_TYPE_VALUES = ["", "New", "Returning"]
ORDER_SOURCE_VALUES = ["", "App", "Web", "Phone"]


# ------------- Auth / Workspace bootstrap -------------
def _make_ws_for_app() -> WorkspaceClient:
    host = DATABRICKS_HOST
    cid = os.getenv("DATABRICKS_CLIENT_ID")
    csec = os.getenv("DATABRICKS_CLIENT_SECRET")

    if not host or not host.startswith("http"):
        raise RuntimeError("DATABRICKS_HOST missing/invalid (must include https://...)")
    if not (cid and csec):
        raise RuntimeError("Service principal creds not found (DATABRICKS_CLIENT_ID/SECRET).")

    cfg = SDKConfig(
        host=host,
        auth_type="oauth-m2m",     # force SP path
        client_id=cid,
        client_secret=csec,
        # Azure would also need oauth_token_endpoint; AWS doesn't
    )
    return WorkspaceClient(config=cfg)

@st.cache_resource(show_spinner=True)
def get_workspace() -> WorkspaceClient:
    return _make_ws_for_app()

WS = get_workspace()


# ------------- Vector Search helper -------------
class VectorSearchManager:
    def __init__(self, ws=None, endpoint_name=None, index_name=None, columns=None):
        # Try new-style (supports workspace=) then fall back to legacy constructor
        try:
            self.client = VectorSearchClient(workspace=ws, disable_notice=True)  # newer SDKs
        except TypeError:
            # Older SDK: relies on env vars (DATABRICKS_HOST, DATABRICKS_AUTH_TYPE, CLIENT_ID/SECRET or TOKEN)
            self.client = VectorSearchClient(disable_notice=True)

        self.endpoint = endpoint_name
        self.index = index_name
        self.columns = columns or []
        self._handle = None

    def _get_index(self):
        if self._handle is None:
            self._handle = self.client.get_index(endpoint_name=self.endpoint, index_name=self.index)
        return self._handle

    @staticmethod
    def _expr(satisfaction: Optional[str], service_method: Optional[str],
              customer_type: Optional[str], order_source: Optional[str]) -> Optional[str]:
        parts = []
        if satisfaction:    parts.append(f"satisfaction = '{satisfaction}'")
        if service_method:  parts.append(f"service_method = '{service_method}'")
        if customer_type:   parts.append(f"customer_type = '{customer_type}'")
        if order_source:    parts.append(f"order_source = '{order_source}'")
        return " AND ".join(parts) if parts else None

    def search(self, query_text: str, top_k: int = 25,
               satisfaction: Optional[str] = None,
               service_method: Optional[str] = None,
               customer_type: Optional[str] = None,
               order_source: Optional[str] = None) -> List[Dict[str, Any]]:
        idx = self._get_index()
        expr = self._expr(satisfaction, service_method, customer_type, order_source)
        params: Dict[str, Any] = {"query_text": query_text, "columns": self.columns, "num_results": max(1, int(top_k))}
        if expr:
            params["filters"] = {"expr": expr}
        try:
            res = idx.similarity_search(**params) or {}
            result = res.get("result", {})
            rows = result.get("data_array", []) or []
            cols = result.get("columns", self.columns)
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            log.error("Vector search failed: %s", e, exc_info=True)
            return []


# ------------- LLM call (optional) -------------
def call_llm(prompt: str, system: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 512) -> str:
    if not LLM_ENDPOINT:
        raise RuntimeError("LLM_ENDPOINT not set.")
    client = get_deploy_client("databricks")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.predict(endpoint=LLM_ENDPOINT, inputs={"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"]
        if "outputs" in resp and resp["outputs"]:
            return resp["outputs"][0].get("content", "")
    return json.dumps(resp)


# ------------- Header / env checks -------------
st.title("🍕 Pizza VOC Explorer")

with st.expander("Auth env check"):
    st.write({
        "HOST set": bool(DATABRICKS_HOST),
        "CLIENT_ID set": bool(os.getenv("DATABRICKS_CLIENT_ID")),
        "CLIENT_SECRET set": bool(os.getenv("DATABRICKS_CLIENT_SECRET")),
        "AUTH_TYPE": os.getenv("DATABRICKS_AUTH_TYPE"),
    })
with st.expander("App env check"):
    st.write({
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "LLM_ENDPOINT": LLM_ENDPOINT or "(unset)",
    })

# Prove Workspace auth now (fail fast if wrong)
with st.expander("Workspace auth probe"):
    try:
        me = WS.current_user.me()
        st.success(f"Authenticated as: {me.user_name}")
    except Exception as e:
        st.error(f"WorkspaceClient failed: {e}")
        st.stop()

# Build Vector Search manager
try:
    VSM = VectorSearchManager(
        ws=WS,
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
        columns=SEARCH_COLUMNS,
    )
except Exception as e:
    st.error(f"Failed to initialize RAG system: {e}")
    st.stop()


# ------------- Search UI -------------
st.subheader("Find and summarize customer feedback")

query = st.text_input("Search phrase", "late delivery cold pizza")
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1:
    satisfaction = st.selectbox("Satisfaction", SATISFACTION_VALUES)
with c2:
    service_method = st.selectbox("Service method", SERVICE_METHOD_VALUES)
with c3:
    customer_type = st.selectbox("Customer type", CUSTOMER_TYPE_VALUES)
with c4:
    order_source = st.selectbox("Order source", ORDER_SOURCE_VALUES)
with c5:
    top_k = st.number_input("Top K", min_value=1, max_value=200, value=25, step=1)

if st.button("Search"):
    with st.spinner("Searching Vector Index…"):
        rows = VSM.search(
            query_text=query,
            top_k=top_k,
            satisfaction=satisfaction or None,
            service_method=service_method or None,
            customer_type=customer_type or None,
            order_source=order_source or None,
        )
    st.write(f"Found {len(rows)} results")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if LLM_ENDPOINT:
            try:
                preview = df[["text", "satisfaction", "service_method"]].head(10).to_dict(orient="records")
                with st.spinner("Summarizing with LLM…"):
                    summary = call_llm(
                        prompt=f"Summarize key customer pain points and themes from these reviews: {preview}",
                        system="You are an analyst who writes concise bullet summaries for restaurant VOC."
                    )
                st.subheader("LLM Summary")
                st.write(summary)
            except Exception as e:
                st.warning(f"LLM call failed: {e}")
    else:
        st.info("No results. Try broadening your filters or removing them.")