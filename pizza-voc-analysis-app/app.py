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
    """Initialize the RAG system components."""
    vector_manager = VectorSearchManager()
    rag_pipeline = RAGPipeline(vector_manager)
    return vector_manager, rag_pipeline

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
    
    # Initialize RAG system
    try:
        vector_manager, rag_pipeline = initialize_rag_system()
        st.success("✅ RAG system initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.stop()
    
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
        with st.spinner("Analyzing customer feedback..."):
            response = rag_pipeline.ask_question(question, num_results, satisfaction_filter)
        
        if response["status"] == "success":
            st.success(f"✅ Found {response['contexts_found']} relevant customer comments")
            
            # Show satisfaction distribution
            if response["metadata"]:
                satisfactions = [meta["satisfaction"] for meta in response["metadata"]]
                satisfaction_counts = pd.Series(satisfactions).value_counts()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Satisfaction Distribution")
                    fig = px.pie(
                        values=satisfaction_counts.values,
                        names=satisfaction_counts.index,
                        title="Feedback by Satisfaction Level"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Service Method Breakdown")
                    service_methods = [meta.get("service_method", "Unknown") for meta in response["metadata"]]
                    service_counts = pd.Series(service_methods).value_counts()
                    
                    fig = px.bar(
                        x=service_counts.index,
                        y=service_counts.values,
                        title="Feedback by Service Method"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
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
