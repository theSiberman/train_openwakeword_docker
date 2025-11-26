#!/bin/bash
# Quick test of Groq TTS API to diagnose issues

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "Run: export GROQ_API_KEY='your-key-here'"
    exit 1
fi

echo "Testing Groq TTS API..."
echo "API Key: ${GROQ_API_KEY:0:10}... (truncated)"
echo ""

# Test API call
echo "Making test request..."
curl -v -X POST "https://api.groq.com/openai/v1/audio/speech" \
    -H "Authorization: Bearer $GROQ_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "playai-tts",
        "input": "test",
        "voice": "Arista-PlayAI",
        "response_format": "wav"
    }' \
    --output test_output.wav

echo ""
echo "Response saved to: test_output.wav"

if [ -f test_output.wav ]; then
    SIZE=$(stat -f%z test_output.wav 2>/dev/null || stat -c%s test_output.wav 2>/dev/null)
    echo "File size: $SIZE bytes"

    if [ $SIZE -lt 1000 ]; then
        echo "⚠️  File is very small - likely an error response"
        echo "Contents:"
        cat test_output.wav
    else
        echo "✓ File looks valid (audio data)"
        file test_output.wav 2>/dev/null || echo "(file command not available)"
    fi
else
    echo "❌ No output file created"
fi
