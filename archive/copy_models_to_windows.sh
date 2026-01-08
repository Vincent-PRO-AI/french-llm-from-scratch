#!/bin/bash
# Copy models to Windows path C:\Users\vincent\code\french-llm-from-scratch\models

SOURCE_DIR="/home/vincent/code/repo/french-llm-from-scratch/models"
# Windows path converted to WSL format
DEST_DIR="/mnt/c/Users/vincent/code/french-llm-from-scratch/models"

echo ""
echo "=================================="
echo "📤 COPYING MODELS TO WINDOWS"
echo "=================================="
echo ""

if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Source not found: $SOURCE_DIR"
    exit 1
fi

# Check if Windows path is accessible
if [ ! -d "/mnt/c" ]; then
    echo "⚠️  WSL Windows mount not available (/mnt/c)"
    echo "    Please copy manually:"
    echo "    From (Linux): $SOURCE_DIR"
    echo "    To (Windows): C:\Users\vincent\code\french-llm-from-scratch\models"
    exit 1
fi

echo "📁 Source: $SOURCE_DIR"
echo "📍 Destination: $DEST_DIR"
echo ""

# Create destination directory
mkdir -p "$DEST_DIR"

echo "⏳ Copying files..."
cp -rv "$SOURCE_DIR"/* "$DEST_DIR/" 2>&1 | tail -20

echo ""
echo "✅ Copy complete!"
echo ""
ls -lh "$DEST_DIR/" | head -10

echo ""
echo "📊 Size:"
du -sh "$DEST_DIR/"
echo ""
