# 🍕 Pizza Company VOC Analysis - Databricks App

An AI-powered customer feedback analysis application using RAG (Retrieval-Augmented Generation) and Databricks Vector Search.

## 📋 Overview

This Streamlit application provides business users with an intuitive interface to analyze customer feedback using advanced AI techniques. It combines semantic search with large language models to deliver actionable insights from customer voice-of-customer (VOC) data.

### ✨ Features

- **💬 Natural Language Queries**: Ask questions about customer feedback in plain English
- **📊 Satisfaction Analysis**: Analyze trends across different satisfaction levels  
- **🔍 Semantic Search**: Find relevant feedback using AI-powered similarity search
- **📈 Executive Dashboard**: Pre-built analytics for business stakeholders
- **🎯 Real-time Insights**: Instant analysis of 6,000+ customer comments
- **🔧 Interactive Filters**: Filter by satisfaction level, service method, and more

## 🏗️ Architecture

```
Customer Feedback (CSV) 
    ↓
Unity Catalog (Delta Tables)
    ↓  
Text Chunking & Preprocessing
    ↓
Databricks Vector Search (BGE Embeddings)
    ↓
RAG Pipeline (Retrieval + Generation)
    ↓
Streamlit App (Interactive Interface)
```

## 📁 Project Structure

```
pizza-voc-analysis-app/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── utils.py              # Utility classes and functions
├── requirements.txt      # Python dependencies
├── databricks.yml        # Databricks App configuration
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

- **Databricks Workspace** with Unity Catalog enabled
- **Vector Search Endpoint**: `dbdemos_vs_endpoint` 
- **Vector Index**: `users.kevin_ippen.voc_chunks_index`
- **DBR 14.3 LTS ML** or higher

### 1. Clone and Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd pizza-voc-analysis-app

# Verify your configuration in config.py
# Update VECTOR_SEARCH_ENDPOINT and VECTOR_INDEX_NAME if needed
```

### 2. Deploy to Databricks

#### Option A: Using Databricks CLI (Recommended)

```bash
# Install Databricks CLI if not already installed
pip install databricks-cli

# Configure authentication
databricks configure

# Deploy the app
databricks bundle deploy --target development
```

#### Option B: Manual Deployment

1. **Upload Files**: Upload all files to a Databricks workspace folder
2. **Create App**: 
   - Go to "Databricks Apps" in your workspace
   - Click "Create App"
   - Select "Upload files" and choose your folder
   - Set entry point to `app.py`
3. **Configure**: Set compute size and environment variables as needed

### 3. Access Your App

Once deployed, your app will be available at:
```
https://your-workspace.databricks.com/apps/your-app-name
```

## ⚙️ Configuration

### Core Settings (config.py)

```python
# Vector Search Configuration
VECTOR_SEARCH_ENDPOINT = "dbdemos_vs_endpoint"
VECTOR_INDEX_NAME = "users.kevin_ippen.voc_chunks_index"

# App Settings  
DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 20
```

### Environment Variables

Set these in your Databricks App configuration:

- `STREAMLIT_SERVER_PORT`: "8501"
- `STREAMLIT_SERVER_ADDRESS`: "0.0.0.0" 
- `STREAMLIT_LOGGER_LEVEL`: "info"

## 🎯 Usage Guide

### 1. Ask Questions Mode
- Enter natural language questions about customer feedback
- Examples:
  - "What do customers complain about most?"
  - "How satisfied are customers with delivery?"
  - "What pizza quality issues need attention?"

### 2. Satisfaction Analysis Mode
- Analyze specific topics across satisfaction levels
- Compare how different customer segments discuss the same topic
- Identify satisfaction-specific trends and patterns

### 3. Search Feedback Mode
- Perform targeted searches with filters
- Find specific types of feedback
- Export results for further analysis

### 4. Business Dashboard Mode
- Executive-level insights and KPIs
- Pre-built analytics for common business questions
- Quick topic analysis with visualizations

## 📊 Sample Business Questions

The app can answer questions like:

**Quality & Product**
- "What pizza quality issues do customers report?"
- "How do customers rate our new menu items?"
- "What ingredients get the most complaints?"

**Service & Operations**  
- "What delivery problems need immediate attention?"
- "How is our customer service rated?"
- "What issues occur with mobile ordering?"

**Customer Satisfaction**
- "What makes customers highly satisfied?"
- "Main reasons for customer dissatisfaction?"
- "How can we improve customer experience?"

## 🔧 Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally (requires Databricks connection)
streamlit run app.py
```

### Code Structure

- **`app.py`**: Main Streamlit interface with multiple analysis modes
- **`utils.py`**: Core classes for vector search and RAG pipeline
- **`config.py`**: Centralized configuration management

### Key Classes

- **`VectorSearchManager`**: Handles all vector search operations
- **`RAGPipeline`**: Manages retrieval-augmented generation workflow
- **`DataValidator`**: Validates inputs and search results

## 🛠️ Troubleshooting

### Common Issues

**1. Vector Search Connection Failed**
```
Solution: Verify VECTOR_SEARCH_ENDPOINT and VECTOR_INDEX_NAME in config.py
Check that the vector index exists and is online
```

**2. No Search Results Found**
```
Solution: Try broader search terms
Check satisfaction level filters
Verify vector index has data
```

**3. App Deployment Failed**
```
Solution: Check databricks.yml configuration
Verify workspace permissions
Ensure all dependencies are in requirements.txt
```

### Debug Mode

Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### Best Practices

1. **Caching**: Streamlit automatically caches vector search initialization
2. **Query Optimization**: Use specific search terms for better results
3. **Result Limits**: Keep num_results reasonable (5-20) for responsive UI
4. **Compute Resources**: Use SMALL compute for development, MEDIUM+ for production

### Monitoring

Monitor app performance via:
- Databricks App metrics dashboard
- Streamlit built-in performance monitoring
- Vector search endpoint metrics

## 🔒 Security & Governance

### Data Protection
- All customer data remains in Unity Catalog
- Vector search uses Databricks-managed embeddings
- No data leaves the Databricks environment

### Access Control
- Configure user permissions in databricks.yml
- Use service principals for production deployments
- Implement row-level security if needed

### Compliance
- Audit logging through Unity Catalog
- Data lineage tracking
- GDPR/CCPA compliance through Databricks governance

## 🚀 Production Deployment

### Checklist

- [ ] Update configuration for production vector index
- [ ] Set up service principal authentication
- [ ] Configure appropriate compute resources
- [ ] Set up monitoring and alerting
- [ ] Test with production data volume
- [ ] Configure backup and disaster recovery

### Scaling Considerations

- **High Traffic**: Use LARGE compute resources
- **Multiple Teams**: Deploy separate instances per team
- **Data Growth**: Monitor vector index performance
- **Global Usage**: Consider multi-region deployment

## 📚 Additional Resources

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Databricks Apps Guide](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)
- [Unity Catalog Best Practices](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please:
1. Check this README and troubleshooting section
2. Review Databricks documentation
3. Contact your Databricks administrator
4. Create an issue in this repository

---

**Built with ❤️ using Databricks, Vector Search, and Streamlit**
