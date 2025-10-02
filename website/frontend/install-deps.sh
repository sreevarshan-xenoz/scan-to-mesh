#!/bin/bash

echo "🦷 Installing OpenDentalScan Website Dependencies..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Make sure you're in the frontend directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "🚀 To start development server:"
    echo "  npm run dev"
    echo ""
    echo "Then visit: http://localhost:3000"
else
    echo "❌ Failed to install dependencies"
    echo "Try running: npm install --legacy-peer-deps"
fi