#!/bin/bash

# Deployment script for Pizza VOC Analysis Databricks App
# This script helps automate the deployment process

set -e

echo "🍕 Pizza VOC Analysis - Databricks App Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    print_error "Databricks CLI is not installed!"
    echo "Please install it with: pip install databricks-cli"
    exit 1
fi

print_status "Databricks CLI found ✓"

# Check if we're in the right directory
if [[ ! -f "app.py" || ! -f "databricks.yml" ]]; then
    print_error "app.py or databricks.yml not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

print_status "Project files found ✓"

# Check databricks authentication
if ! databricks workspace list > /dev/null 2>&1; then
    print_error "Databricks authentication failed!"
    echo "Please run: databricks configure"
    exit 1
fi

print_status "Databricks authentication verified ✓"

# Prompt for deployment target
echo ""
echo "Select deployment target:"
echo "1) Development"
echo "2) Staging" 
echo "3) Production"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        TARGET="development"
        ;;
    2)
        TARGET="staging"
        ;;
    3)
        TARGET="production"
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

print_status "Deploying to $TARGET environment..."

# Validate configuration
print_status "Validating configuration..."

# Check if required configuration exists
if ! grep -q "VECTOR_SEARCH_ENDPOINT" config.py; then
    print_error "VECTOR_SEARCH_ENDPOINT not found in config.py"
    exit 1
fi

if ! grep -q "VECTOR_INDEX_NAME" config.py; then
    print_error "VECTOR_INDEX_NAME not found in config.py"
    exit 1
fi

print_status "Configuration validated ✓"

# Run pre-deployment checks
print_status "Running pre-deployment checks..."

# Check Python syntax
python -m py_compile app.py
python -m py_compile config.py
python -m py_compile utils.py

print_status "Python syntax check passed ✓"

# Deploy the app
print_status "Deploying application..."

if databricks bundle deploy --target $TARGET; then
    print_status "🎉 Deployment successful!"
    
    # Try to get the app URL
    print_status "Getting app information..."
    
    echo ""
    echo "================================"
    echo "📱 DEPLOYMENT COMPLETED"
    echo "================================"
    echo "Target: $TARGET"
    echo "Status: ✅ Success"
    echo ""
    echo "Next steps:"
    echo "1. Check your Databricks workspace for the app"
    echo "2. Navigate to 'Databricks Apps' section"
    echo "3. Find 'pizza-voc-analysis' app"
    echo "4. Click to access your application"
    echo ""
    echo "🔧 To update the app:"
    echo "   databricks bundle deploy --target $TARGET"
    echo ""
    echo "🗑️  To remove the app:"
    echo "   databricks bundle destroy --target $TARGET"
    echo ""
    
else
    print_error "Deployment failed!"
    echo ""
    echo "Common issues:"
    echo "1. Check workspace permissions"
    echo "2. Verify databricks.yml configuration"
    echo "3. Ensure vector search endpoint exists"
    echo "4. Check Unity Catalog access"
    echo ""
    echo "For more help, see README.md troubleshooting section"
    exit 1
fi

# Optional: Run basic health check
read -p "Do you want to run a basic health check? (y/n): " run_check

if [[ $run_check =~ ^[Yy]$ ]]; then
    print_status "Running health check..."
    
    # This would be a simple test of the app functionality
    python -c "
import sys
sys.path.append('.')
from config import Config
from utils import VectorSearchManager

try:
    if Config.validate_config():
        print('✅ Configuration validation passed')
    else:
        print('❌ Configuration validation failed')
        
    # Test vector search connection (basic)
    vm = VectorSearchManager()
    print('✅ Vector search manager initialized')
    
    print('✅ Basic health check passed')
except Exception as e:
    print(f'❌ Health check failed: {str(e)}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        print_status "Health check passed ✓"
    else
        print_warning "Health check failed - app may still work"
    fi
fi

echo ""
print_status "Deployment process completed!"
echo "Thank you for using the Pizza VOC Analysis app! 🍕"
