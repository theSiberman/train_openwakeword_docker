#!/bin/bash
# Generate training samples using Groq's TTS API (19 English voices!)
# Requires: GROQ_API_KEY environment variable
#
# Usage: ./generate_groq_samples.sh [--confusables|--positives|--negatives] [--count 100]
#
# Options:
#   --confusables  Generate phonetically similar phrases (for negatives)
#   --positives    Generate wake word with multiple voices (for positives)
#   --negatives    Generate common responses with multiple voices (for negatives)
#   --count N      Number of samples to generate (default: 100)

set -e

MODE="confusables"
COUNT=100
WAKE_WORD="hey zaadoz"
OUTPUT_DIR="./generated_samples"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --confusables)
            MODE="confusables"
            shift
            ;;
        --positives)
            MODE="positives"
            shift
            ;;
        --negatives)
            MODE="negatives"
            shift
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --wake-word)
            WAKE_WORD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for API key
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY environment variable not set"
    echo ""
    echo "Get your API key from: https://console.groq.com/keys"
    echo "Then run: export GROQ_API_KEY='your-key-here'"
    exit 1
fi

echo "========================================"
echo "Groq TTS Sample Generation"
echo "========================================"
echo "Mode: $MODE"
echo "Count: $COUNT samples"
echo "Wake word: $WAKE_WORD"
echo ""

# Groq TTS API - 19 English voices!
GROQ_VOICES=(
    "Arista-PlayAI"
    "Atlas-PlayAI"
    "Basil-PlayAI"
    "Briggs-PlayAI"
    "Calum-PlayAI"
    "Celeste-PlayAI"
    "Cheyenne-PlayAI"
    "Chip-PlayAI"
    "Cillian-PlayAI"
    "Deedee-PlayAI"
    "Fritz-PlayAI"
    "Gail-PlayAI"
    "Indigo-PlayAI"
    "Mamaw-PlayAI"
    "Mason-PlayAI"
    "Mikail-PlayAI"
    "Mitch-PlayAI"
    "Quinn-PlayAI"
    "Thunder-PlayAI"
)

# Phonetically similar phrases to wake word (confusables)
CONFUSABLE_PHRASES=(
    "they say those"
    "hey let's go"
    "they're crazy"
    "hey sarah knows"
    "they say so"
    "hey the radios"
    "they may go"
    "hey what a dose"
    "they play shows"
    "hey chaos"
    "they raise those"
    "hey amazing"
    "they say no"
    "hey say hello"
    "they gaze so"
    "hey gracious"
    "they phase out"
    "hey day close"
    "they blaze through"
    "hey erase those"
)

# Common TTS responses
RESPONSE_PHRASES=(
    "Okay"
    "Done"
    "Sure"
    "Got it"
    "Right away"
    "On it"
    "Turning on the lights"
    "Turning off the lights"
    "Setting temperature"
    "Playing music"
    "Volume adjusted"
    "Timer started"
    "Alarm set"
    "Understood"
    "Of course"
    "One moment"
    "All set"
    "Complete"
    "Light is on"
    "Light is off"
)

# Function to generate TTS using Groq API
generate_tts() {
    local text="$1"
    local voice="$2"
    local output_file="$3"

    curl -s -X POST "https://api.groq.com/openai/v1/audio/speech" \
        -H "Authorization: Bearer $GROQ_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"playai-tts\",
            \"input\": \"$text\",
            \"voice\": \"$voice\",
            \"response_format\": \"wav\"
        }" \
        --output "$output_file"

    # Convert to 16kHz mono if needed (openwakeword requirement)
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -i "$output_file" -ar 16000 -ac 1 -y "${output_file%.wav}_16k.wav" 2>/dev/null
        mv "${output_file%.wav}_16k.wav" "$output_file"
    fi
}

mkdir -p "$OUTPUT_DIR"
mkdir -p "./negative_samples"

GENERATED=0

case $MODE in
    confusables)
        echo "🎯 Generating confusable negatives (prevent false positives)..."
        echo "   Using ${#GROQ_VOICES[@]} different voices"
        echo ""

        OUTPUT_DIR="./negative_samples"

        for ((i=0; i<$COUNT; i++)); do
            # Cycle through phrases and voices
            PHRASE_IDX=$((i % ${#CONFUSABLE_PHRASES[@]}))
            VOICE_IDX=$((i % ${#GROQ_VOICES[@]}))

            PHRASE="${CONFUSABLE_PHRASES[$PHRASE_IDX]}"
            VOICE="${GROQ_VOICES[$VOICE_IDX]}"

            OUTPUT_FILE="$OUTPUT_DIR/groq_confusable_$(printf '%04d' $i).wav"

            echo -n "  [$((i+1))/$COUNT] \"$PHRASE\" ($VOICE)... "

            if generate_tts "$PHRASE" "$VOICE" "$OUTPUT_FILE"; then
                echo "✓"
                GENERATED=$((GENERATED + 1))
            else
                echo "✗ (failed)"
            fi

            # Rate limiting - be nice to API
            sleep 0.1
        done
        ;;

    positives)
        echo "🎤 Generating positive samples with voice variety..."
        echo "   Using ${#GROQ_VOICES[@]} different voices"
        echo ""

        OUTPUT_DIR="./generated_samples"

        for ((i=0; i<$COUNT; i++)); do
            VOICE_IDX=$((i % ${#GROQ_VOICES[@]}))
            VOICE="${GROQ_VOICES[$VOICE_IDX]}"

            OUTPUT_FILE="$OUTPUT_DIR/groq_positive_$(printf '%04d' $i).wav"

            echo -n "  [$((i+1))/$COUNT] \"$WAKE_WORD\" ($VOICE)... "

            if generate_tts "$WAKE_WORD" "$VOICE" "$OUTPUT_FILE"; then
                echo "✓"
                GENERATED=$((GENERATED + 1))
            else
                echo "✗ (failed)"
            fi

            sleep 0.1
        done
        ;;

    negatives)
        echo "🔇 Generating TTS response negatives (prevent feedback loop)..."
        echo "   Using ${#GROQ_VOICES[@]} different voices"
        echo ""

        OUTPUT_DIR="./negative_samples"

        for ((i=0; i<$COUNT; i++)); do
            PHRASE_IDX=$((i % ${#RESPONSE_PHRASES[@]}))
            VOICE_IDX=$((i % ${#GROQ_VOICES[@]}))

            PHRASE="${RESPONSE_PHRASES[$PHRASE_IDX]}"
            VOICE="${GROQ_VOICES[$VOICE_IDX]}"

            OUTPUT_FILE="$OUTPUT_DIR/groq_response_$(printf '%04d' $i).wav"

            echo -n "  [$((i+1))/$COUNT] \"$PHRASE\" ($VOICE)... "

            if generate_tts "$PHRASE" "$VOICE" "$OUTPUT_FILE"; then
                echo "✓"
                GENERATED=$((GENERATED + 1))
            else
                echo "✗ (failed)"
            fi

            sleep 0.1
        done
        ;;
esac

echo ""
echo "✅ Generated $GENERATED samples using Groq TTS"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
