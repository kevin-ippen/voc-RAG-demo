"""
Utility classes and functions for the Pizza VOC Analysis Databricks App.
"""

from databricks.vector_search.client import VectorSearchClient
from typing import Dict, List, Any, Optional
import logging

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearchManager:
    """Manages vector search operations for customer feedback."""
    
    def __init__(self):
        """Initialize the vector search manager."""
        self.client = VectorSearchClient(disable_notice=True)
        self.endpoint_name = Config.VECTOR_SEARCH_ENDPOINT
        self.index_name = Config.VECTOR_INDEX_NAME
        self.search_columns = Config.SEARCH_COLUMNS
        
        # Validate configuration
        if not Config.validate_config():
            raise ValueError("Invalid configuration. Please check your settings.")
    
    def _get_index(self):
        """Get the vector search index."""
        try:
            return self.client.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
        except Exception as e:
            logger.error(f"Failed to get vector index: {str(e)}")
            raise
    
    def search_feedback(self, query: str, num_results: int = 5, satisfaction_filter: Optional[str] = None) -> List[List]:
        """
        Search customer feedback using vector similarity.
        
        Args:
            query: Search query text
            num_results: Number of results to return
            satisfaction_filter: Optional filter by satisfaction level
            
        Returns:
            List of search results [id, text, satisfaction, service_method, score]
        """
        try:
            index = self._get_index()
            
            # Build search parameters
            search_params = {
                "query_text": query,
                "columns": self.search_columns,
                "num_results": min(num_results, Config.MAX_NUM_RESULTS)
            }
            
            # Add satisfaction filter if specified
            if satisfaction_filter and satisfaction_filter in Config.SATISFACTION_LEVELS:
                search_params["filters"] = {"satisfaction": satisfaction_filter}
            
            # Perform search
            results = index.similarity_search(**search_params)
            
            # Extract and return data array
            if results and 'result' in results and 'data_array' in results['result']:
                return results['result']['data_array']
            
            return []
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return []
    
    def analyze_satisfaction_trends(self, topic: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Analyze how different satisfaction levels discuss a topic.
        
        Args:
            topic: Topic to analyze
            num_results: Number of results per satisfaction level
            
        Returns:
            Dictionary with satisfaction analysis results
        """
        try:
            results = {}
            total_found = 0
            
            for satisfaction_level in Config.SATISFACTION_LEVELS:
                feedback = self.search_feedback(
                    query=topic,
                    num_results=num_results,
                    satisfaction_filter=satisfaction_level
                )
                
                # Format results for easier consumption
                formatted_feedback = []
                for result in feedback:
                    formatted_feedback.append({
                        "id": result[0],
                        "text": result[1],
                        "satisfaction": result[2],
                        "service_method": result[3],
                        "score": result[4] if len(result) > 4 else 0.0
                    })
                
                results[satisfaction_level] = formatted_feedback
                total_found += len(formatted_feedback)
            
            return {
                "topic": topic,
                "satisfaction_analysis": results,
                "total_found": total_found
            }
            
        except Exception as e:
            logger.error(f"Satisfaction trend analysis failed for topic '{topic}': {str(e)}")
            return {
                "topic": topic,
                "satisfaction_analysis": {},
                "total_found": 0,
                "error": str(e)
            }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the searchable data."""
        try:
            # Sample search to get basic info
            sample_results = self.search_feedback("pizza", num_results=100)
            
            if sample_results:
                satisfactions = [result[2] for result in sample_results]
                service_methods = [result[3] for result in sample_results]
                
                return {
                    "total_sample_size": len(sample_results),
                    "satisfaction_distribution": {
                        satisfaction: satisfactions.count(satisfaction)
                        for satisfaction in set(satisfactions)
                    },
                    "service_method_distribution": {
                        method: service_methods.count(method)
                        for method in set(service_methods)
                    }
                }
            
            return {"error": "No data available"}
            
        except Exception as e:
            logger.error(f"Failed to get search statistics: {str(e)}")
            return {"error": str(e)}

class RAGPipeline:
    """Manages the complete RAG (Retrieval-Augmented Generation) pipeline."""
    
    def __init__(self, vector_manager: VectorSearchManager):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_manager: VectorSearchManager instance
        """
        self.vector_manager = vector_manager
        self.prompt_template = Config.RAG_PROMPT_TEMPLATE
    
    def create_rag_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Create a RAG prompt for the LLM.
        
        Args:
            question: User's question
            contexts: List of relevant context strings
            
        Returns:
            Formatted RAG prompt
        """
        # Format contexts with numbers
        context_text = "\n\n".join([
            f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        ])
        
        return self.prompt_template.format(
            context=context_text,
            question=question
        )
    
    def ask_question(self, question: str, num_contexts: int = 5, satisfaction_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant contexts and prepare for generation.
        
        Args:
            question: User's question about VOC data
            num_contexts: Number of context chunks to retrieve
            satisfaction_filter: Optional satisfaction level filter
            
        Returns:
            Dictionary with question, contexts, metadata, and RAG prompt
        """
        try:
            # Step 1: Retrieve relevant contexts
            search_results = self.vector_manager.search_feedback(
                query=question,
                num_results=num_contexts,
                satisfaction_filter=satisfaction_filter
            )
            
            # Step 2: Extract contexts and metadata
            contexts = []
            metadata = []
            
            for result in search_results:
                # result format: [id, text, satisfaction, service_method, score]
                contexts.append(result[1])  # text
                metadata.append({
                    "id": result[0],
                    "satisfaction": result[2],
                    "service_method": result[3],
                    "score": result[4] if len(result) > 4 else 0.0
                })
            
            # Step 3: Create RAG prompt
            rag_prompt = self.create_rag_prompt(question, contexts)
            
            # Step 4: Return structured response
            return {
                "question": question,
                "contexts_found": len(contexts),
                "contexts": contexts,
                "metadata": metadata,
                "rag_prompt": rag_prompt,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed for question '{question}': {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "status": "error"
            }
    
    def generate_business_insights(self, topic: str, num_contexts: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive business insights for a specific topic.
        
        Args:
            topic: Business topic to analyze
            num_contexts: Number of contexts to analyze
            
        Returns:
            Structured business insights
        """
        try:
            # Get satisfaction trend analysis
            trend_analysis = self.vector_manager.analyze_satisfaction_trends(topic, num_contexts//len(Config.SATISFACTION_LEVELS))
            
            # Generate summary insights for each satisfaction level
            insights = {}
            for satisfaction_level in Config.SATISFACTION_LEVELS:
                comments = trend_analysis["satisfaction_analysis"].get(satisfaction_level, [])
                
                if comments:
                    # Extract key themes (simplified keyword analysis)
                    all_text = " ".join([comment["text"] for comment in comments])
                    
                    insights[satisfaction_level] = {
                        "comment_count": len(comments),
                        "sample_comments": [comment["text"][:100] + "..." for comment in comments[:3]],
                        "avg_relevance_score": sum(comment["score"] for comment in comments) / len(comments),
                        "service_method_breakdown": self._analyze_service_methods(comments)
                    }
            
            return {
                "topic": topic,
                "insights": insights,
                "total_comments_analyzed": trend_analysis["total_found"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Business insights generation failed for topic '{topic}': {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "status": "error"
            }
    
    def _analyze_service_methods(self, comments: List[Dict]) -> Dict[str, int]:
        """Analyze service method distribution in comments."""
        service_methods = [comment.get("service_method", "Unknown") for comment in comments]
        return {method: service_methods.count(method) for method in set(service_methods)}

class DataValidator:
    """Validates data and search results."""
    
    @staticmethod
    def validate_search_query(query: str) -> bool:
        """Validate search query."""
        return bool(query and query.strip() and len(query.strip()) >= 2)
    
    @staticmethod
    def validate_search_results(results: List) -> bool:
        """Validate search results format."""
        if not results:
            return True  # Empty results are valid
        
        # Check if first result has expected structure
        if results and len(results[0]) >= 4:
            return True
        
        return False
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize user input query."""
        if not query:
            return ""
        
        # Basic sanitization
        sanitized = query.strip()
        # Remove any potentially harmful characters
        sanitized = ''.join(char for char in sanitized if char.isprintable())
        
        return sanitized[:200]  # Limit length

# Utility functions
def format_satisfaction_distribution(metadata: List[Dict]) -> Dict[str, int]:
    """Format satisfaction distribution from metadata."""
    satisfactions = [meta.get("satisfaction", "Unknown") for meta in metadata]
    return {satisfaction: satisfactions.count(satisfaction) for satisfaction in set(satisfactions)}

def format_service_method_distribution(metadata: List[Dict]) -> Dict[str, int]:
    """Format service method distribution from metadata."""
    service_methods = [meta.get("service_method", "Unknown") for meta in metadata]
    return {method: service_methods.count(method) for method in set(service_methods)}

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."