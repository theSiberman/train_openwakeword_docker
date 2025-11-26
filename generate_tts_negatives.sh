#!/bin/bash
# Generate negative samples using Alan Medium voice (your TTS voice)
# This teaches the model NOT to trigger on TTS responses
#
# Usage: ./generate_tts_negatives.sh [count] [--force]
#
# Options:
#   count    Number of TTS negative samples to generate (default: 500)
#   --force  Skip prompts and regenerate all TTS samples

set -e

VOICE_DIR="./models"
NEGATIVE_DIR="./negative_samples"
TTS_VOICE="$VOICE_DIR/en_GB-alan-medium.onnx"
TTS_NEGATIVE_COUNT=${1:-500}  # Default 500 TTS negative samples
FORCE_REGENERATE=false

# Parse optional arguments
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_REGENERATE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "TTS Negative Sample Generation"
echo "========================================"
echo "Voice: Alan Medium (your TTS voice)"
echo "Count: $TTS_NEGATIVE_COUNT samples"
echo ""

# Common TTS response phrases that might trigger false positives
TTS_PHRASES=(
    "Okay"
    "Done"
    "Turning on the lights"
    "Turning off the lights"
    "Setting temperature to 20 degrees"
    "Playing music"
    "Stopping playback"
    "Setting volume to 50 percent"
    "The time is 3:45 PM"
    "The weather is sunny"
    "I didn't understand that"
    "Sorry, I can't do that"
    "Alarm set for 7 AM"
    "Timer started"
    "Reminder added"
    "Light is now on"
    "Light is now off"
    "Temperature set"
    "Volume adjusted"
    "Music is playing"
    "Media paused"
    "All lights off"
    "Good morning"
    "Good night"
    "Yes"
    "No"
    "Sure"
    "Of course"
    "Right away"
    "One moment"
    "Just a second"
    "I'm on it"
    "Command executed"
    "Task completed"
    "Understood"
    "Got it"
    "Opening curtains"
    "Closing curtains"
    "Activating scene"
    "Running automation"
    "Brightness set to 75 percent"
    "Color changed to blue"
    "Front door is locked"
    "Garage door is closed"
    "Security system armed"
    "Motion detected"
    "Doorbell pressed"
    "Temperature is 22 degrees"
    "Humidity is 60 percent"
    "Air quality is good"
    "Battery level is low"
)

if [ ! -f "$TTS_VOICE" ]; then
    echo "❌ Error: Alan Medium voice not found at $TTS_VOICE"
    echo "   Run ./setup_direct.sh first"
    exit 1
fi

mkdir -p "$NEGATIVE_DIR"

# --- Check for existing TTS negative samples ---
EXISTING_TTS=$(ls "$NEGATIVE_DIR"/tts_alan_*.wav 2>/dev/null | wc -l)

if [ $EXISTING_TTS -gt 0 ]; then
    echo "⚠️  Existing TTS negative samples detected:"
    echo "   • Found $EXISTING_TTS TTS samples (tts_alan_*.wav) in $NEGATIVE_DIR"
    echo ""

    if [ "$FORCE_REGENERATE" = true ]; then
        echo "🗑️  --force flag detected, deleting and regenerating TTS samples..."
        rm -f "$NEGATIVE_DIR"/tts_alan_*.wav
        echo "✓ Deleted. Starting fresh TTS generation..."
        echo ""
    else
        echo "Options:"
        echo "  1) Skip generation (use existing TTS samples)"
        echo "  2) Delete and regenerate TTS samples"
        echo "  3) Keep existing and add more TTS samples"
        echo "  4) Cancel"
        echo ""
        read -p "Choose [1-4]: " choice

        case $choice in
            1)
                echo "✓ Skipping TTS generation, using $EXISTING_TTS existing samples"
                echo ""
                echo "Ready to train!"
                exit 0
                ;;
            2)
                echo "🗑️  Deleting existing TTS samples..."
                rm -f "$NEGATIVE_DIR"/tts_alan_*.wav
                echo "✓ Deleted. Starting fresh TTS generation..."
                echo ""
                ;;
            3)
                echo "✓ Keeping existing $EXISTING_TTS samples, will add more"
                echo ""
                ;;
            4)
                echo "Cancelled."
                exit 0
                ;;
            *)
                echo "Invalid choice. Cancelled."
                exit 1
                ;;
        esac
    fi
fi

# Determine Python command
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

echo "Generating TTS negative samples..."
echo ""

# Calculate how many times to repeat phrases to reach target count
NUM_PHRASES=${#TTS_PHRASES[@]}
REPEATS=$(( (TTS_NEGATIVE_COUNT + NUM_PHRASES - 1) / NUM_PHRASES ))

echo "  → Using ${NUM_PHRASES} unique phrases"
echo "  → Repeating ${REPEATS}x with variations"
echo ""

GENERATED=0
TEMP_DIR="$NEGATIVE_DIR/tts_temp"
mkdir -p "$TEMP_DIR"

for ((rep=0; rep<$REPEATS; rep++)); do
    for phrase in "${TTS_PHRASES[@]}"; do
        if [ $GENERATED -ge $TTS_NEGATIVE_COUNT ]; then
            break 2
        fi

        # Generate with acoustic variations
        OMP_NUM_THREADS=4 ORT_NUM_THREADS=4 $PYTHON_CMD /opt/piper-sample-generator/generate_samples.py "$phrase" \
            --model "$TTS_VOICE" \
            --max-samples 1 \
            --output-dir "$TEMP_DIR" \
            --slerp-weights 0.3 \
            --length-scales 0.8 1.0 1.2 \
            --noise-scales 0.4 0.667 0.8 2>&1 | grep -v "WARNING\|pthread_setaffinity_np" || true

        # Rename and move
        for file in "$TEMP_DIR"/*.wav; do
            if [ -f "$file" ]; then
                mv "$file" "$NEGATIVE_DIR/tts_alan_${GENERATED}.wav"
                GENERATED=$((GENERATED + 1))
            fi
        done

        if (( (GENERATED % 50) == 0 )); then
            echo "  → Generated $GENERATED/$TTS_NEGATIVE_COUNT TTS negative samples..."
        fi
    done
done

rmdir "$TEMP_DIR" 2>/dev/null || true

echo ""
echo "✓ Generated $GENERATED TTS negative samples using Alan Medium voice"
echo ""
echo "These samples teach the model NOT to trigger on TTS responses"
echo ""
