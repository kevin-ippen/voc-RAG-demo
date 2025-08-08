# app.py
import os
import json
import logging
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd

# ----- Databricks clients -----
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from databricks import sql
from databricks.sdk.core import Config

# ------------- Streamlit setup -------------
st.set_page_config(
    page_title="Pizza VOC Explorer",
    layout="wide",
    page_icon="🍕",
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pizza-voc-app")

# ------------- Environment / Config -------------
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # required for OBO and SDKs
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT")  # e.g., 'dbdemos_vs_endpoint'
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "users.kevin_ippen.voc_pizza_chunks_for_index")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")  # e.g., 'databricks-gpt-oss-20b'
SQL_WAREHOUSE_HTTP_PATH = os.getenv("SQL_WAREHOUSE_HTTP_PATH", "/sql/1.0/warehouses/REPLACE_ME")

# Index schema (as provided)
SEARCH_COLUMNS: List[str] = [
    "id",
    "text",
    "text_with_context",
    "source_record_id",
    "satisfaction",
    "service_method",
    "customer_type",
    "order_source",
    "order_date",
    "service_time",
    "chunk_length",
    "chunk_word_count",
]

# ------------- Helpers -------------
def _normalize_nullable(s: Optional[str]) -> Optional[str]:
    """Return None for empty/whitespace strings; strip otherwise."""
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None

def _make_filter_expr(
    satisfaction: Optional[str] = None,
    service_method: Optional[str] = None,
    customer_type: Optional[str] = None,
    order_source: Optional[str] = None,
) -> Optional[str]:
    """
    Build a Vector Search 'expr' filter matching your schema.
    Assumes exact, case-sensitive matches on string columns.
    """
    exprs = []
    if satisfaction:
        exprs.append(f"satisfaction = '{satisfaction}'")
    if service_method:
        exprs.append(f"service_method = '{service_method}'")
    if customer_type:
        exprs.append(f"customer_type = '{customer_type}'")
    if order_source:
        exprs.append(f"order_source = '{order_source}'")
    return " AND ".join(exprs) if exprs else None


# ------------- Clients / RAG Init -------------
@st.cache_resource(show_spinner=True)
def initialize_clients():
    """
    Initialize clients for Vector Search + MLflow Deployments and validate basics.
    Raises with helpful messages if required envs are missing.
    """
    if not DATABRICKS_HOST:
        raise RuntimeError("DATABRICKS_HOST is not set. Add it in app env.")

    if not VECTOR_SEARCH_ENDPOINT:
        raise RuntimeError("VECTOR_SEARCH_ENDPOINT is not set. Add it in app env.")
    if not VECTOR_INDEX_NAME:
        raise RuntimeError("VECTOR_INDEX_NAME is not set. Add it in app env.")

    # Vector Search
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
    )

    # LLM (optional – we keep app working even if not configured)
    deploy_client = None
    if LLM_ENDPOINT:
        try:
            deploy_client = get_deploy_client("databricks")
        except Exception as e:
            log.warning("Failed to init MLflow deployments client: %s", e)

    return {"vsc": vsc, "index": index, "deploy_client": deploy_client}


def call_llm(prompt: str, system_message: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Calls a Databricks model serving endpoint via MLflow Deployments with a chat-style payload.
    Requires LLM_ENDPOINT env var to be set to the serving endpoint name.
    """
    if not LLM_ENDPOINT:
        raise RuntimeError("LLM_ENDPOINT not configured in env; cannot call LLM.")

    client = get_deploy_client("databricks")
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = client.predict(endpoint=LLM_ENDPOINT, inputs=payload)

    # Common response shapes
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"]
        if "outputs" in resp and resp["outputs"]:
            return resp["outputs"][0].get("content", "")
    return json.dumps(resp)


def vs_search(index, query_text: str, top_k: int,
              satisfaction: Optional[str], service_method: Optional[str],
              customer_type: Optional[str], order_source: Optional[str]) -> List[Dict]:
    """Vector Search similarity with optional expr filter; returns list of dict rows."""
    expr = _make_filter_expr(
        _normalize_nullable(satisfaction),
        _normalize_nullable(service_method),
        _normalize_nullable(customer_type),
        _normalize_nullable(order_source),
    )
    filters = {"expr": expr} if expr else None

    results = index.similarity_search(
        query_text=query_text,
        columns=SEARCH_COLUMNS,
        num_results=top_k,
        filters=filters
    )
    rows = results.get("result", {}).get("data_array", [])
    cols = results.get("result", {}).get("columns", SEARCH_COLUMNS)
    return [dict(zip(cols, r)) for r in rows]


# ------------- UI: Header -------------
st.title("🍕 Pizza VOC Explorer")

with st.expander("Environment check (safe to share)"):
    st.write({
        "DATABRICKS_HOST set": bool(DATABRICKS_HOST),
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT or "(unset)",
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME or "(unset)",
        "LLM_ENDPOINT": LLM_ENDPOINT or "(unset)",
        "SQL_WAREHOUSE_HTTP_PATH": SQL_WAREHOUSE_HTTP_PATH or "(unset)",
    })

# ------------- RAG / Search Panel -------------
st.subheader("Find and summarize customer feedback")

query = st.text_input("Search phrase", "late delivery cold pizza")
col_a, col_b, col_c, col_d, col_k = st.columns([1,1,1,1,1])
with col_a:
    satisfaction = st.selectbox("Satisfaction", ["", "Satisfied", "Neutral", "Dissatisfied"])
with col_b:
    service_method = st.selectbox("Service method", ["", "Delivery", "Pickup"])
with col_c:
    customer_type = st.selectbox("Customer type", ["", "New", "Returning"])
with col_d:
    order_source = st.selectbox("Order source", ["", "App", "Web", "Phone"])
with col_k:
    top_k = st.number_input("Top K", min_value=1, max_value=200, value=25, step=1)

search_btn = st.button("Search")

clients = None
init_error = None
try:
    clients = initialize_clients()
except Exception as e:
    init_error = e

if init_error:
    st.error(f"Failed to initialize RAG system: {init_error}")
elif search_btn:
    try:
        with st.spinner("Searching feedback..."):
            results = vs_search(
                clients["index"],
                query,
                top_k,
                satisfaction,
                service_method,
                customer_type,
                order_source
            )
        st.success(f"Found {len(results)} results")
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            if clients["deploy_client"] and LLM_ENDPOINT:
                try:
                    with st.spinner("Summarizing with LLM..."):
                        preview = df[["text", "satisfaction", "service_method"]].head(10).to_dict(orient="records")
                        summary = call_llm(
                            prompt=f"Summarize key customer pain points and themes from these reviews: {preview}",
                            system_message="You are an analyst who writes concise bullet summaries for restaurant VOC."
                        )
                    st.subheader("LLM Summary")
                    st.write(summary)
                except Exception as e:
                    st.warning(f"LLM call failed: {e}")
            else:
                st.info("LLM_ENDPOINT not configured; skipping summary.")
        else:
            st.info("No results. Try broadening your filters or removing them.")
    except Exception as e:
        st.exception(e)


# ------------- OBO SQL Warehouse Test -------------
st.divider()
st.subheader("SQL Warehouse OBO Test (viewer identity)")

cfg = Config()  # reads DATABRICKS_HOST from env, no PAT/SP needed for OBO

def _get_obo_token() -> Optional[str]:
    try:
        headers = st.context.headers  # only available behind Databricks Apps proxy
        return headers.get("X-Forwarded-Access-Token")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def connect_with_obo(server_hostname: str, http_path: str, user_token: str):
    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=user_token,
    )

table_name = st.text_input("Table to sample", "samples.nyctaxi.trips")

if st.button("Run OBO Query"):
    if not (cfg.host or DATABRICKS_HOST):
        st.error("DATABRICKS_HOST not set. Add it to app env.")
        st.stop()
    server_hostname = cfg.host or DATABRICKS_HOST

    if not SQL_WAREHOUSE_HTTP_PATH or "REPLACE_ME" in SQL_WAREHOUSE_HTTP_PATH:
        st.error("Set SQL_WAREHOUSE_HTTP_PATH in env to your actual warehouse HTTP path.")
        st.stop()

    token = _get_obo_token()
    if not token:
        st.error("No OBO token found. Open this app inside Databricks and ensure you are signed in.")
        st.stop()

    try:
        conn = connect_with_obo(server_hostname, SQL_WAREHOUSE_HTTP_PATH, token)
        with conn.cursor() as c:
            c.execute(f"SELECT current_user() AS user, current_catalog() AS catalog, current_schema() AS schema")
            who = c.fetchall_arrow().to_pandas()
        st.caption("Session identity")
        st.dataframe(who, use_container_width=True)

        with conn.cursor() as c:
            c.execute(f"SELECT * FROM {table_name} LIMIT 10")
            df = c.fetchall_arrow().to_pandas()
        st.caption("Sample rows")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.exception(e)
