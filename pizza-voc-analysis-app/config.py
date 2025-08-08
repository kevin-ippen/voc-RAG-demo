"""
Configuration settings for the Pizza VOC Analysis Databricks App.
"""

class Config:
    """Configuration class containing all app settings."""
    
    # Vector Search Configuration
    VECTOR_SEARCH_ENDPOINT = "dbdemos_vs_endpoint"
    VECTOR_INDEX_NAME = "users.kevin_ippen.voc_chunks_index"
    
    # Model Serving Configuration
    LLM_ENDPOINT_NAME = "databricks-llama-2-70b-chat"  # Update with your actual LLM endpoint
    # Alternative endpoints you might have:
    # LLM_ENDPOINT_NAME = "databricks-dbrx-instruct"
    # LLM_ENDPOINT_NAME = "your-custom-llm-endpoint"
    
    # Search Configuration
    DEFAULT_NUM_RESULTS = 5
    MAX_NUM_RESULTS = 20
    
    # LLM Configuration
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500
    
    # Available satisfaction levels
    SATISFACTION_LEVELS = [
        "Highly Satisfied",
        "Satisfied", 
        "Not Satisfied",
        "Accuracy",
        "Taste",
        "Appearance"
    ]
    
    # Required columns for vector search
    SEARCH_COLUMNS = ["id", "text", "satisfaction", "service_method"]
    
    # App Settings
    APP_TITLE = "🍕 Pizza Company VOC Analysis"
    APP_DESCRIPTION = "AI-Powered Customer Feedback Insights using RAG and Vector Search"
    
    # Business question templates
    SAMPLE_QUESTIONS = [
        "What do customers complain about most?",
        "How satisfied are customers with delivery?",
        "What do customers love about our pizza?",
        "Are there issues with order accuracy?", 
        "How is our customer service rated?",
        "What problems occur with mobile ordering?",
        "What do customers say about wait times?",
        "Quality issues mentioned by customers?",
        "What makes customers highly satisfied?",
        "Common problems with carryout orders?",
        "How do customers rate our pizza taste?",
        "Issues with online ordering system?"
    ]
    
    # Analysis topics for quick insights
    QUICK_ANALYSIS_TOPICS = [
        "pizza quality",
        "delivery service",
        "customer service", 
        "order accuracy",
        "wait times",
        "mobile app",
        "website ordering",
        "store cleanliness",
        "staff friendliness",
        "value for money"
    ]
    
    # RAG prompt template
    RAG_PROMPT_TEMPLATE = """You are a helpful assistant analyzing customer feedback for a pizza company. 
Based on the customer comments provided below, answer the question accurately and provide insights.

Customer Feedback Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided customer feedback
- Provide specific examples from the comments when relevant
- If the context doesn't contain enough information, say so
- Identify patterns and trends in customer sentiment when applicable
- Be concise but informative
- Focus on actionable insights for business improvement

Answer:"""

    # Visualization settings
    CHART_COLORS = {
        "Highly Satisfied": "#2E8B57",    # Sea Green
        "Satisfied": "#90EE90",           # Light Green
        "Not Satisfied": "#FF6347",       # Tomato Red
        "Accuracy": "#FFD700",            # Gold
        "Taste": "#FF69B4",               # Hot Pink
        "Appearance": "#87CEEB"           # Sky Blue
    }
    
    # Error messages
    ERROR_MESSAGES = {
        "vector_search_init": "Failed to initialize vector search. Please check your configuration.",
        "search_failed": "Search failed. Please try a different query.",
        "no_results": "No matching customer comments found. Try different search terms.",
        "rag_failed": "Failed to generate insights. Please try again.",
        "llm_failed": "Failed to generate AI response. Please check model serving endpoint."
    }
    
    @classmethod
    def get_chart_color(cls, satisfaction_level: str) -> str:
        """Get chart color for satisfaction level."""
        return cls.CHART_COLORS.get(satisfaction_level, "#808080")  # Default gray
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        required_settings = [
            cls.VECTOR_SEARCH_ENDPOINT,
            cls.VECTOR_INDEX_NAME,
            cls.SEARCH_COLUMNS,
            cls.LLM_ENDPOINT_NAME
        ]
        
        return all(setting for setting in required_settings)