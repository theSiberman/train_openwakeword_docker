#!/bin/bash
# Interactive script to record wake word samples
# Records 16kHz mono WAV files suitable for training

set -e

WAKE_WORD="${1:-hey zaadoz}"
OUTPUT_DIR="./real_samples"
START_NUM=${2:-442}  # Start after existing samples

mkdir -p "$OUTPUT_DIR"

# Check if arecord is available
if ! command -v arecord &> /dev/null; then
    echo "❌ Error: arecord not found"
    echo "   Install: sudo apt-get install alsa-utils"
    exit 1
fi

echo "========================================"
echo "Wake Word Sample Recording"
echo "========================================"
echo "Wake word: $WAKE_WORD"
echo "Output: $OUTPUT_DIR"
echo "Starting at: sample_$(printf '%04d' $START_NUM)"
echo ""
echo "Recording tips:"
echo "  • Vary your distance from mic (near/far)"
echo "  • Vary your volume (whisper/normal/loud)"
echo "  • Vary background noise (quiet/music/talking)"
echo "  • Record at different times (morning/evening)"
echo "  • Say it naturally - don't over-enunciate!"
echo ""
echo "Press ENTER to record, Ctrl+C to quit"
echo "========================================"
echo ""

SAMPLE_NUM=$START_NUM

while true; do
    read -p "Sample $SAMPLE_NUM - Ready? "

    echo "  🔴 Recording in 3..."
    sleep 1
    echo "  🔴 2..."
    sleep 1
    echo "  🔴 1..."
    sleep 1
    echo "  🎤 SAY: '$WAKE_WORD' NOW!"

    OUTPUT_FILE="$OUTPUT_DIR/real_sample_$(printf '%04d' $SAMPLE_NUM).wav"

    # Record 3 seconds, 16kHz, mono, 16-bit
    arecord -d 3 -f S16_LE -r 16000 -c 1 -q "$OUTPUT_FILE" 2>/dev/null

    # Get file size
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

    echo "  ✓ Saved: $OUTPUT_FILE ($SIZE)"
    echo ""

    SAMPLE_NUM=$((SAMPLE_NUM + 1))

    # Show progress every 50 samples
    if (( (SAMPLE_NUM - START_NUM) % 50 == 0 )); then
        TOTAL=$((SAMPLE_NUM - START_NUM))
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Progress: $TOTAL samples recorded!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
done
