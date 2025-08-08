"""
Pizza Company VOC Analysis - Databricks App
A Streamlit application for analyzing customer feedback using RAG and vector search.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks.vector_search.client import VectorSearchClient
from typing import Dict, List, Any
import time

from config import Config
from utils import VectorSearchManager, RAGPipeline

# Page configuration
st.set_page_config(
    page_title="🍕 Pizza VOC Analysis",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components with detailed error handling."""
    try:
        # Test vector search connection first
        from databricks.vector_search.client import VectorSearchClient
        
        client = VectorSearchClient(disable_notice=True)
        
        # Try to get the index
        index = client.get_index(
            endpoint_name=Config.VECTOR_SEARCH_ENDPOINT,
            index_name=Config.VECTOR_INDEX_NAME
        )
        
        # Test a simple search to verify it's working
        test_results = index.similarity_search(
            query_text="test",
            columns=Config.SEARCH_COLUMNS,
            num_results=1
        )
        
        # Test model serving connection
        from databricks.sdk import WorkspaceClient
        workspace_client = WorkspaceClient()
        
        try:
            # Test model serving endpoint
            test_response = workspace_client.serving_endpoints.query(
                name=Config.LLM_ENDPOINT_NAME,
                prompt="Test connection",
                temperature=0.1,
                max_tokens=10
            )
            model_serving_available = True
        except Exception as model_error:
            print(f"Model serving test failed: {model_error}")
            model_serving_available = False
        
        # If we get here, vector search is working
        vector_manager = VectorSearchManager()
        rag_pipeline = RAGPipeline(vector_manager)
        
        return vector_manager, rag_pipeline, model_serving_available
        
    except Exception as e:
        # Provide more specific error information
        error_msg = str(e)
        
        if "does not exist" in error_msg.lower():
            raise Exception(f"Vector index '{Config.VECTOR_INDEX_NAME}' not found. Please check the index name.")
        elif "endpoint" in error_msg.lower():
            raise Exception(f"Vector search endpoint '{Config.VECTOR_SEARCH_ENDPOINT}' not accessible. Please check the endpoint name.")
        elif "permission" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise Exception("Permission denied. Please check Unity Catalog access permissions.")
        elif "columns" in error_msg.lower():
            raise Exception("Column mismatch. Please verify the vector index schema matches expected columns.")
        else:
            raise Exception(f"Initialization failed: {error_msg}")


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🍕 Pizza Company VOC Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Customer Feedback Insights")
    
    # Initialize RAG system with proper error handling
    vector_manager = None
    rag_pipeline = None
    model_serving_available = False
    initialization_success = False
    
    try:
        vector_manager, rag_pipeline, model_serving_available = initialize_rag_system()
        st.success("✅ RAG system initialized successfully!")
        
        if model_serving_available:
            st.success("✅ Model serving endpoint accessible!")
        else:
            st.warning("⚠️ Model serving endpoint not accessible. AI insights will be limited.")
            
        initialization_success = True
        
        # Store model serving status in session state
        st.session_state.model_serving_available = model_serving_available
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.error("Please check your configuration and try again.")
        st.markdown("""
        **Common issues:**
        - Vector search endpoint not accessible
        - Vector index not found or not ready
        - Model serving endpoint not found
        - Authentication/permission issues
        
        **To fix:**
        1. Verify your vector search endpoint is running
        2. Check that the vector index exists and is online
        3. Verify model serving endpoint exists and has proper permissions
        4. Ensure proper Unity Catalog permissions
        """)
        # Don't stop completely, show the error state
        initialization_success = False
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Analysis mode
        analysis_mode = st.selectbox(
            "Choose Analysis Mode:",
            ["💬 Ask Questions", "📊 Satisfaction Analysis", "🔍 Search Feedback", "📈 Business Dashboard"]
        )
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        num_results = st.slider("Number of results:", 1, 20, 5)
        
        satisfaction_filter = st.selectbox(
            "Filter by satisfaction:",
            ["All", "Highly Satisfied", "Satisfied", "Not Satisfied", "Accuracy", "Taste", "Appearance"]
        )
        
        if satisfaction_filter == "All":
            satisfaction_filter = None
        
        # LLM settings
        model_serving_available = st.session_state.get('model_serving_available', False)
        
        if model_serving_available:
            use_llm = st.checkbox("🤖 Generate AI Insights", value=True, help="Use LLM to generate insights from customer feedback")
            st.session_state.use_llm = use_llm
            
            if use_llm:
                with st.expander("🔧 LLM Settings"):
                    temperature = st.slider("Response creativity:", 0.0, 1.0, Config.LLM_TEMPERATURE, 0.1, 
                                          help="Higher values make responses more creative")
                    max_tokens = st.slider("Response length:", 100, 1000, Config.LLM_MAX_TOKENS, 50,
                                         help="Maximum length of AI response")
                    
                    # Store in session state for use in main app
                    st.session_state.llm_temperature = temperature
                    st.session_state.llm_max_tokens = max_tokens
            else:
                st.session_state.llm_temperature = Config.LLM_TEMPERATURE
                st.session_state.llm_max_tokens = Config.LLM_MAX_TOKENS
        else:
            st.info("🤖 AI insights unavailable (model serving endpoint not accessible)")
            st.session_state.use_llm = False
            st.session_state.llm_temperature = Config.LLM_TEMPERATURE
            st.session_state.llm_max_tokens = Config.LLM_MAX_TOKENS
        
        st.divider()
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "What do customers complain about most?",
            "How satisfied are customers with delivery?",
            "What do customers love about our pizza?",
            "Are there issues with order accuracy?",
            "How is our customer service rated?",
            "What problems occur with mobile ordering?",
            "What do customers say about wait times?",
            "Quality issues mentioned by customers?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.selected_question = question
    
    # Main content based on mode
    if not initialization_success:
        # Configuration help if initialization failed
        st.warning("⚠️ RAG system not available. Please fix the configuration issues above.")
        
        # Add debug information
        with st.expander("🔍 Debug Information"):
            st.markdown("**Current Configuration:**")
            st.code(f"""
Vector Search Endpoint: {Config.VECTOR_SEARCH_ENDPOINT}
Vector Index Name: {Config.VECTOR_INDEX_NAME}
Search Columns: {Config.SEARCH_COLUMNS}
            """)
            
            # Show validation results
            st.markdown("**Configuration Validation:**")
            if Config.validate_config():
                st.success("✅ Configuration format is valid")
            else:
                st.error("❌ Configuration format is invalid")
            
            # Quick connection test button
            if st.button("🧪 Test Connection"):
                with st.spinner("Testing vector search connection..."):
                    try:
                        from databricks.vector_search.client import VectorSearchClient
                        client = VectorSearchClient(disable_notice=True)
                        
                        # Test endpoint
                        try:
                            endpoint = client.get_endpoint(Config.VECTOR_SEARCH_ENDPOINT)
                            st.success(f"✅ Endpoint '{Config.VECTOR_SEARCH_ENDPOINT}' accessible")
                        except Exception as e:
                            st.error(f"❌ Endpoint error: {str(e)}")
                            
                        # Test index
                        try:
                            index = client.get_index(
                                endpoint_name=Config.VECTOR_SEARCH_ENDPOINT,
                                index_name=Config.VECTOR_INDEX_NAME
                            )
                            st.success(f"✅ Index '{Config.VECTOR_INDEX_NAME}' accessible")
                            
                            # Test search
                            results = index.similarity_search(
                                query_text="test",
                                columns=Config.SEARCH_COLUMNS,
                                num_results=1
                            )
                            st.success("✅ Search functionality working")
                            
                        except Exception as e:
                            st.error(f"❌ Index error: {str(e)}")
                            
                    except Exception as e:
                        st.error(f"❌ General connection error: {str(e)}")
        
        st.subheader("🔧 Troubleshooting Guide")
        st.markdown("""
        **Step 1: Verify Vector Search Setup**
        - Check that your vector search endpoint is running
        - Ensure the vector index exists and is online
        - Verify the index name matches exactly
        
        **Step 2: Test Access Permissions**
        ```python
        from databricks.vector_search.client import VectorSearchClient
        client = VectorSearchClient(disable_notice=True)
        
        # Test endpoint
        endpoint = client.get_endpoint("dbdemos_vs_endpoint")
        print(f"Endpoint status: {endpoint.endpoint_status}")
        
        # Test index
        index = client.get_index(
            endpoint_name="dbdemos_vs_endpoint",
            index_name="users.kevin_ippen.voc_chunks_index"
        )
        print("Index accessible!")
        ```
        
        **Step 3: Common Solutions**
        - **Index not found**: Check the exact index name in your workspace
        - **Permission denied**: Ensure Unity Catalog access permissions
        - **Endpoint offline**: Wait for endpoint to come online or restart it
        - **Column mismatch**: Verify the index schema matches expected columns
        """)
        
        return  # Exit early if not initialized
    
    # Normal operation if initialization succeeded
    if analysis_mode == "💬 Ask Questions":
        show_question_interface(rag_pipeline, num_results, satisfaction_filter)
    
    elif analysis_mode == "📊 Satisfaction Analysis":
        show_satisfaction_analysis(vector_manager, num_results)
    
    elif analysis_mode == "🔍 Search Feedback":
        show_search_interface(vector_manager, num_results, satisfaction_filter)
    
    elif analysis_mode == "📈 Business Dashboard":
        show_business_dashboard(vector_manager, rag_pipeline)

def show_question_interface(rag_pipeline: RAGPipeline, num_results: int, satisfaction_filter: str):
    """Show the question-answering interface."""
    
    st.header("💬 Ask Questions About Customer Feedback")
    
    # Defensive check
    if rag_pipeline is None:
        st.error("❌ RAG pipeline not available. Please check the configuration.")
        return
    
    # Question input
    default_question = st.session_state.get('selected_question', '')
    question = st.text_input(
        "Enter your question:",
        value=default_question,
        placeholder="e.g., What are customers saying about delivery times?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    if ask_button and question:
        # Get LLM settings from session state
        use_llm = st.session_state.get('model_serving_available', False) and st.session_state.get('use_llm', True)
        
        with st.spinner("Analyzing customer feedback..."):
            try:
                response = rag_pipeline.ask_question(question, num_results, satisfaction_filter, use_llm=use_llm)
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                return
        
        if response["status"] == "success":
            st.success(f"✅ Found {response['contexts_found']} relevant customer comments")
            
            # Show AI-generated insights if available
            if response.get("llm_response") and use_llm:
                st.subheader("🤖 AI-Generated Insights")
                with st.container():
                    st.markdown(response["llm_response"])
                    
                st.divider()
            
            # Show satisfaction distribution
            if response["metadata"]:
                satisfactions = [meta["satisfaction"] for meta in response["metadata"]]
                satisfaction_counts = pd.Series(satisfactions).value_counts()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Satisfaction Distribution")
                    try:
                        fig = px.pie(
                            values=satisfaction_counts.values,
                            names=satisfaction_counts.index,
                            title="Feedback by Satisfaction Level"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating satisfaction chart: {str(e)}")
                
                with col2:
                    st.subheader("📈 Service Method Breakdown")
                    try:
                        service_methods = [meta.get("service_method", "Unknown") for meta in response["metadata"]]
                        service_counts = pd.Series(service_methods).value_counts()
                        
                        fig = px.bar(
                            x=service_counts.index,
                            y=service_counts.values,
                            title="Feedback by Service Method"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating service method chart: {str(e)}")
            
            # Show relevant contexts
            st.subheader("🎯 Relevant Customer Comments")
            for i, (context, meta) in enumerate(zip(response["contexts"], response["metadata"])):
                with st.expander(f"Comment {i+1} - {meta['satisfaction']} ({meta.get('service_method', 'Unknown')})"):
                    st.write(context)
                    if 'score' in meta:
                        st.caption(f"Relevance Score: {meta['score']:.3f}")
            
            # Show RAG prompt (for debugging/transparency)
            with st.expander("🔧 See AI Analysis Prompt"):
                st.code(response["rag_prompt"], language="text")
        
        else:
            st.error(f"❌ Error: {response['error']}")
            
            # Provide some troubleshooting help
            st.markdown("""
            **Possible issues:**
            - Vector search connection problem
            - No matching customer feedback found
            - Query processing error
            
            Try:
            - Using different search terms
            - Removing satisfaction filters
            - Checking the system status above
            """)

def show_satisfaction_analysis(vector_manager: VectorSearchManager, num_results: int):
    """Show satisfaction trend analysis."""
    
    st.header("📊 Satisfaction Trend Analysis")
    
    topic = st.text_input(
        "Enter topic to analyze:",
        placeholder="e.g., pizza quality, delivery service, customer service"
    )
    
    if st.button("📈 Analyze Trends", type="primary") and topic:
        with st.spinner(f"Analyzing '{topic}' across satisfaction levels..."):
            analysis = vector_manager.analyze_satisfaction_trends(topic, num_results)
        
        if analysis["total_found"] > 0:
            st.success(f"✅ Found {analysis['total_found']} relevant comments about '{topic}'")
            
            # Create tabs for each satisfaction level
            satisfaction_levels = ["Highly Satisfied", "Satisfied", "Not Satisfied", "Accuracy", "Taste", "Appearance"]
            tabs = st.tabs(satisfaction_levels)
            
            for i, (satisfaction, tab) in enumerate(zip(satisfaction_levels, tabs)):
                with tab:
                    comments = analysis["satisfaction_analysis"].get(satisfaction, [])
                    
                    if comments:
                        st.write(f"**{len(comments)} comments found**")
                        
                        for j, comment in enumerate(comments):
                            st.write(f"{j+1}. {comment['text']}")
                            if 'score' in comment:
                                st.caption(f"Relevance: {comment['score']:.3f}")
                            st.divider()
                    else:
                        st.info("No relevant comments found for this satisfaction level.")
            
            # Summary visualization
            st.subheader("📊 Distribution Summary")
            summary_data = {
                satisfaction: len(comments) 
                for satisfaction, comments in analysis["satisfaction_analysis"].items()
                if comments
            }
            
            if summary_data:
                fig = px.bar(
                    x=list(summary_data.keys()),
                    y=list(summary_data.values()),
                    title=f"Number of '{topic}' Comments by Satisfaction Level",
                    labels={"x": "Satisfaction Level", "y": "Number of Comments"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"No comments found related to '{topic}'. Try a different topic.")

def show_search_interface(vector_manager: VectorSearchManager, num_results: int, satisfaction_filter: str):
    """Show the search interface."""
    
    st.header("🔍 Search Customer Feedback")
    
    search_query = st.text_input(
        "Search for specific feedback:",
        placeholder="e.g., delivery problems, great service, pizza quality"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    if search_button and search_query:
        with st.spinner("Searching customer feedback..."):
            results = vector_manager.search_feedback(search_query, num_results, satisfaction_filter)
        
        if results:
            st.success(f"✅ Found {len(results)} matching comments")
            
            for i, result in enumerate(results):
                # result format: [id, text, satisfaction, service_method, score]
                with st.expander(f"Result {i+1} - {result[2]} ({result[3]})"):
                    st.write(result[1])  # text
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Satisfaction", result[2])
                    with col2:
                        st.metric("Service Method", result[3])
                    with col3:
                        if len(result) > 4:
                            st.metric("Relevance Score", f"{result[4]:.3f}")
        else:
            st.warning("No matching comments found. Try a different search term.")

def show_business_dashboard(vector_manager: VectorSearchManager, rag_pipeline: RAGPipeline):
    """Show executive business dashboard."""
    
    st.header("📈 Executive Business Dashboard")
    
    # Key business metrics
    st.subheader("🎯 Key Insights")
    
    business_questions = [
        "What are the top customer complaints?",
        "What do highly satisfied customers praise most?",
        "What delivery issues need immediate attention?",
        "How can we improve pizza quality based on feedback?"
    ]
    
    for question in business_questions:
        with st.expander(f"📋 {question}"):
            if st.button(f"Analyze", key=f"biz_{hash(question)}"):
                with st.spinner("Generating insights..."):
                    response = rag_pipeline.ask_question(question, 10)
                
                if response["status"] == "success":
                    # Show key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Comments Analyzed", response["contexts_found"])
                    
                    with col2:
                        satisfactions = [meta["satisfaction"] for meta in response["metadata"]]
                        most_common = pd.Series(satisfactions).mode()[0] if satisfactions else "N/A"
                        st.metric("Most Common Level", most_common)
                    
                    with col3:
                        service_methods = [meta.get("service_method", "Unknown") for meta in response["metadata"]]
                        service_counts = pd.Series(service_methods).value_counts()
                        if not service_counts.empty:
                            st.metric("Primary Service", service_counts.index[0])
                    
                    # Show sample feedback
                    st.write("**Sample Customer Comments:**")
                    for i, context in enumerate(response["contexts"][:3]):
                        st.write(f"• {context[:150]}...")

    # Quick analytics
    st.subheader("⚡ Quick Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🍕 Pizza Quality Analysis"):
            with st.spinner("Analyzing pizza quality feedback..."):
                analysis = vector_manager.analyze_satisfaction_trends("pizza quality", 5)
                
                # Show quick summary
                total_comments = analysis["total_found"]
                st.metric("Total Pizza Quality Comments", total_comments)
                
                if total_comments > 0:
                    summary_data = {
                        satisfaction: len(comments) 
                        for satisfaction, comments in analysis["satisfaction_analysis"].items()
                        if comments
                    }
                    
                    fig = px.pie(
                        values=list(summary_data.values()),
                        names=list(summary_data.keys()),
                        title="Pizza Quality Feedback Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("🚚 Delivery Service Analysis"):
            with st.spinner("Analyzing delivery feedback..."):
                analysis = vector_manager.analyze_satisfaction_trends("delivery service", 5)
                
                total_comments = analysis["total_found"]
                st.metric("Total Delivery Comments", total_comments)
                
                if total_comments > 0:
                    summary_data = {
                        satisfaction: len(comments) 
                        for satisfaction, comments in analysis["satisfaction_analysis"].items()
                        if comments
                    }
                    
                    fig = px.pie(
                        values=list(summary_data.values()),
                        names=list(summary_data.keys()),
                        title="Delivery Service Feedback Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()