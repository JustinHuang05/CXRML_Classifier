#!/bin/bash

# CXR Main App Deployment Script
echo "🚀 Starting deployment process for CXR Main App..."

# Create timestamp for version
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERSION_LABEL="cxr-main-app-${TIMESTAMP}"

echo "📦 Creating deployment package..."

# Clean up any previous deployment files
rm -f cxr-main-app-*.zip

# Create deployment package
zip -r "${VERSION_LABEL}.zip" . \
    -x "*.DS_Store" \
    -x "*.git*" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x "uploads/*" \
    -x ".elasticbeanstalk/*" \
    -x "deploy.sh" \
    -x "cxr-main-app-*.zip"

echo "✅ Deployment package created: ${VERSION_LABEL}.zip"

echo ""
echo "🔧 Deployment Options:"
echo "1. AWS Console Upload:"
echo "   - Go to: https://console.aws.amazon.com/elasticbeanstalk/"
echo "   - Find your 'cxr-main-app' application"
echo "   - Click 'Upload and Deploy'"
echo "   - Upload: ${VERSION_LABEL}.zip"
echo "   - Version Label: ${VERSION_LABEL}"
echo ""
echo "2. CLI Deployment (if permissions are fixed):"
echo "   eb deploy --label ${VERSION_LABEL}"
echo ""
echo "📋 Changes in this deployment:"
echo "   - Added comprehensive error handling for external API calls"
echo "   - Added timeout protection (30 seconds)"
echo "   - Added JSON parsing validation"
echo "   - Added proper logging for debugging"
echo "   - Fixed JSONDecodeError when external service returns HTML"
echo ""
echo "🎯 This should resolve the 503 Server Error issues you were seeing!" 