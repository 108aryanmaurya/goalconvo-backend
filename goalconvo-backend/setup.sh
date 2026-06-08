#!/bin/bash

# GoalConvo Framework Setup Script
# This script will set up the GoalConvo framework for first-time use

set -e  # Exit on any error

echo "🚀 Setting up GoalConvo Framework..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version OK: $python_version"

# Install package in development mode
echo "📦 Installing GoalConvo package..."
pip install -e .

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{multiwoz,synthetic,few_shot_hub,results}
mkdir -p logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before running the framework"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.py

# Test installation
echo "🧪 Testing installation..."
python -c "from goalconvo import Config; print('✅ GoalConvo package imported successfully')"

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Download MultiWOZ dataset:"
echo "   python scripts/download_multiwoz.py"
echo ""
echo "3. Test connection:"
echo "   python scripts/generate_dialogues.py --test-connection"
echo ""
echo "4. Generate dialogues:"
echo "   python scripts/generate_dialogues.py --num-dialogues 100"
echo ""
echo "5. Evaluate results:"
echo "   python scripts/comprehensive_dialogue_evaluation.py"
echo ""
echo "📖 For setup and usage, see ../README.md"
