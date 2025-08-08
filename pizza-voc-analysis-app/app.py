import os
import streamlit as st

from utils import _make_ws, VectorSearchManager

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")

# ---------- Clients (cached) ----------
@st.cache_resource(show_spinner=True)
def get_workspace():
    # Tries SP → PAT → default env; raises with a clear message if all fail
    return _make_ws()

@st.cache_resource(show_spinner=True)
def get_vsm(_ws):
    return VectorSearchManager(ws=_ws)

WS = get_workspace()
VSM = get_vsm(WS)

# ---------- Sidebar config ----------
with st.sidebar:
    st.header("Configuration")
    st.write("**Workspace**:", WS.config.host)
    st.write("**VS endpoint**:", VSM.endpoint_name)
    st.write("**Index**:", VSM.index_fqn)

    st.divider()
    st.caption("Filters (optional)")
    satisfaction = st.selectbox("Satisfaction", ["", "low", "medium", "high"], index=0)
    service_method = st.selectbox("Service Method", ["", "delivery", "carryout", "dine-in"], index=0)
    customer_type = st.selectbox("Customer Type", ["", "new", "returning"], index=0)
    order_source = st.selectbox("Order Source", ["", "web", "mobile", "phone", "pos"], index=0)

# ---------- Main UI ----------
st.title("🔎 RAG Search")

query = st.text_input("Enter a query", placeholder="e.g., late delivery, wrong order, missing item...")
top_k = st.slider("Results", min_value=1, max_value=50, value=10)

col_run, col_clear = st.columns([1, 1])
with col_run:
    run = st.button("Search")
with col_clear:
    if st.button("Clear"):
        st.experimental_rerun()

if run and query.strip():
    with st.spinner("Searching…"):
        try:
            rows = VSM.search(
                query_text=query.strip(),
                top_k=top_k,
                satisfaction=satisfaction or None,
                service_method=service_method or None,
                customer_type=customer_type or None,
                order_source=order_source or None,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not rows:
        st.info("No results.")
    else:
        for i, r in enumerate(rows, start=1):
            with st.container(border=True):
                st.markdown(f"**#{i}**   ·   **Score:** {r.get('score')}")
                if "text" in r:
                    st.write(r["text"])
                meta = {k: v for k, v in r.items() if k not in ("text", "score")}
                if meta:
                    with st.expander("Metadata"):
                        st.json(meta)
