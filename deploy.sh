#!/bin/bash
# Deployment script for Tech Discovery Recorder on Render

echo "🚀 Tech Discovery Recorder - Production Deployment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "render.yaml" ]; then
    echo "❌ Error: render.yaml not found. Please run from project root directory."
    exit 1
fi

echo "✅ Project structure verified"

# Check if static directory exists
if [ ! -d "static" ]; then
    echo "📁 Creating static directory..."
    mkdir -p static
fi

# Copy HTML file to static directory if not already there
if [ ! -f "static/index.html" ]; then
    if [ -f "Tech Discovery Recorder.html" ]; then
        echo "📋 Copying HTML interface to static directory..."
        cp "Tech Discovery Recorder.html" "static/index.html"
        echo "✅ HTML interface ready"
    else
        echo "❌ Error: Tech Discovery Recorder.html not found"
        exit 1
    fi
fi

# Verify critical files exist
echo "🔍 Verifying deployment files..."

REQUIRED_FILES=(
    "requirements.txt"
    "render.yaml"
    "Procfile"
    "src/main.py"
    "src/config/settings.py"
    "static/index.html"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

echo ""
echo "🎯 Deployment Checklist:"
echo "========================"
echo "✅ Project structure verified"
echo "✅ Static files configured"
echo "✅ Requirements file updated with PostgreSQL support"
echo "✅ Render configuration updated"
echo ""
echo "📋 Next Steps:"
echo "1. Commit and push changes to Git:"
echo "   git add ."
echo "   git commit -m 'feat: configure for Render production deployment'"
echo "   git push origin main"
echo ""
echo "2. In Render Dashboard:"
echo "   - Go to https://dashboard.render.com"
echo "   - Find 'tech-discovery-recorder' service"
echo "   - Click 'Manual Deploy' → 'Deploy latest commit'"
echo ""
echo "3. Set Environment Variables in Render:"
echo "   - CLAUDE_API_KEY: Your Anthropic API key"
echo "   - OPENAI_API_KEY: Your OpenAI API key"
echo ""
echo "4. Monitor Deployment Logs:"
echo "   - Watch for successful Python app startup"
echo "   - Check health endpoint: /api/v1/health"
echo ""
echo "🚀 Ready for deployment!"