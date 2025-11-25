#!/bin/bash
# Generate training samples (positive wake word + negative background audio)
# Usage: ./generate_samples_direct.sh "wake word phrase" [--positive-count 3000] [--negative-count 6000]

set -e

# --- Default Configuration ---
WAKE_WORD="${1:-hey zaadoz}"
POSITIVE_COUNT=3000  # Default from OpenWakeWord best practices
NEGATIVE_COUNT=6000  # 2:1 ratio negative to positive
VOICE_DIR="./models"
GENERATED_DIR="./generated_samples"
REAL_DIR="./real_samples"
NEGATIVE_DIR="./negative_samples"

# Parse optional arguments
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --positive-count)
            POSITIVE_COUNT="$2"
            shift 2
            ;;
        --negative-count)
            NEGATIVE_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate wake word
if [ -z "$WAKE_WORD" ]; then
    echo "❌ Error: Wake word phrase is required"
    echo "Usage: $0 \"wake word phrase\" [--positive-count 3000] [--negative-count 6000]"
    exit 1
fi

echo "========================================"
echo "Sample Generation"
echo "========================================"
echo "Wake word: $WAKE_WORD"
echo "Target positive samples: $POSITIVE_COUNT"
echo "Target negative samples: $NEGATIVE_COUNT"
echo ""

# Determine Python command
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

# --- Step 1: Count existing real samples ---
mkdir -p "$REAL_DIR"
REAL_COUNT=$(ls "$REAL_DIR"/*.wav 2>/dev/null | wc -l)
echo "[1/2] Real sample count: $REAL_COUNT"

# Calculate how many synthetic positive samples we need
SYNTHETIC_POSITIVE=$((POSITIVE_COUNT - REAL_COUNT))
if [ $SYNTHETIC_POSITIVE -lt 0 ]; then
    SYNTHETIC_POSITIVE=0
    echo "  ⚠ You have more real samples than needed. Using first $POSITIVE_COUNT samples."
fi

echo "  → Will generate $SYNTHETIC_POSITIVE synthetic positive samples"
echo ""

# --- Step 2: Generate positive samples with variety ---
if [ $SYNTHETIC_POSITIVE -gt 0 ]; then
    echo "[2/2] Generating positive samples with voice variety..."
    mkdir -p "$GENERATED_DIR"

    # Define voice models (4 voices)
    VOICES=(
        "$VOICE_DIR/en_US-lessac-medium.onnx"
        "$VOICE_DIR/en_GB-alan-medium.onnx"
        "$VOICE_DIR/en_GB-alba-medium.onnx"
        "$VOICE_DIR/en_US-libritts-high.onnx"
    )

    # Calculate samples per voice (distribute evenly)
    SAMPLES_PER_VOICE=$((SYNTHETIC_POSITIVE / ${#VOICES[@]}))
    REMAINDER=$((SYNTHETIC_POSITIVE % ${#VOICES[@]}))

    echo "  → Generating ~$SAMPLES_PER_VOICE samples per voice (4 voices total)"
    echo ""

    # Generate samples for each voice with acoustic variations
    for i in "${!VOICES[@]}"; do
        VOICE="${VOICES[$i]}"
        VOICE_NAME=$(basename "$VOICE" .onnx)

        # Add extra sample to first voices if there's a remainder
        if [ $i -lt $REMAINDER ]; then
            COUNT=$((SAMPLES_PER_VOICE + 1))
        else
            COUNT=$SAMPLES_PER_VOICE
        fi

        if [ ! -f "$VOICE" ]; then
            echo "  ⚠ Skipping $VOICE_NAME (model not found)"
            continue
        fi

        echo "  → $VOICE_NAME: generating $COUNT samples..."

        $PYTHON_CMD /opt/piper-sample-generator/generate_samples.py "$WAKE_WORD" \
            --model "$VOICE" \
            --max-samples $COUNT \
            --output-dir "$GENERATED_DIR" \
            --slerp-weights 0.3 \
            --length-scales "0.8,1.0,1.2" \
            --noise-scales "0.4,0.667,0.8" 2>&1 | grep -v "WARNING"
    done

    # Count generated samples
    GENERATED_COUNT=$(ls "$GENERATED_DIR"/*.wav 2>/dev/null | wc -l)
    echo ""
    echo "  ✓ Generated $GENERATED_COUNT synthetic positive samples"
else
    echo "[2/2] Skipping positive sample generation (have enough real samples)"
    GENERATED_COUNT=0
fi

echo ""

# --- Step 3: Download negative samples if needed ---
EXISTING_NEGATIVE=$(ls "$NEGATIVE_DIR"/*.wav 2>/dev/null | wc -l)

if [ $EXISTING_NEGATIVE -ge $NEGATIVE_COUNT ]; then
    echo "✓ Negative samples already exist ($EXISTING_NEGATIVE samples)"
else
    echo "Downloading negative samples (speech, noise, music)..."
    mkdir -p "$NEGATIVE_DIR"

    # Download negative samples using Python script
    $PYTHON_CMD << 'PYTHON_DOWNLOAD'
import os
import sys
from datasets import load_dataset
import soundfile as sf
import numpy as np

negative_dir = "./negative_samples"
target_count = int(os.environ.get("NEGATIVE_COUNT", 6000))

print(f"  → Downloading {target_count} negative samples from datasets...")
print("  → This may take 10-15 minutes on first run...")

# Calculate distribution: 60% speech, 20% noise, 20% music
speech_count = int(target_count * 0.6)  # 3600
noise_count = int(target_count * 0.2)   # 1200
music_count = target_count - speech_count - noise_count  # 1200

count = 0

# 1. Download speech samples from Common Voice
try:
    print(f"  → Downloading {speech_count} speech clips from Common Voice...")
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True, trust_remote_code=True)
    for i, sample in enumerate(dataset):
        if i >= speech_count:
            break
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))

        output_path = os.path.join(negative_dir, f"speech_{i:05d}.wav")
        sf.write(output_path, audio, 16000)
        count += 1
        if (i + 1) % 500 == 0:
            print(f"    Downloaded {i + 1}/{speech_count} speech clips...")
except Exception as e:
    print(f"  ⚠ Error downloading speech: {e}")

# 2. Download noise samples from FSD50k (simplified - just use ESC-50 which is easier)
try:
    print(f"  → Downloading {noise_count} noise clips from ESC-50...")
    dataset = load_dataset("ashraq/esc50", split="train", trust_remote_code=True)
    for i in range(min(noise_count, len(dataset))):
        audio = np.array(dataset[i]["audio"]["array"])
        sr = dataset[i]["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))

        output_path = os.path.join(negative_dir, f"noise_{i:05d}.wav")
        sf.write(output_path, audio, 16000)
        count += 1
        if (i + 1) % 200 == 0:
            print(f"    Downloaded {i + 1}/{noise_count} noise clips...")
except Exception as e:
    print(f"  ⚠ Error downloading noise: {e}")

# 3. Generate music samples from GTZAN (or use FMA free music)
try:
    print(f"  → Downloading {music_count} music clips from GTZAN...")
    dataset = load_dataset("marsyas/gtzan", "all", split="train", trust_remote_code=True)
    for i in range(min(music_count, len(dataset))):
        audio = np.array(dataset[i]["audio"]["array"])
        sr = dataset[i]["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))

        # Take 5-second chunks
        chunk_size = 16000 * 5
        if len(audio) > chunk_size:
            audio = audio[:chunk_size]

        output_path = os.path.join(negative_dir, f"music_{i:05d}.wav")
        sf.write(output_path, audio, 16000)
        count += 1
        if (i + 1) % 200 == 0:
            print(f"    Downloaded {i + 1}/{music_count} music clips...")
except Exception as e:
    print(f"  ⚠ Error downloading music: {e}")

print(f"\n  ✓ Downloaded {count} negative samples total")
PYTHON_DOWNLOAD

    EXISTING_NEGATIVE=$(ls "$NEGATIVE_DIR"/*.wav 2>/dev/null | wc -l)
fi

# --- Summary ---
echo ""
echo "========================================"
echo "Sample Generation Complete!"
echo "========================================"
echo ""
echo "Positive samples:"
echo "  • Real recordings: $REAL_COUNT"
echo "  • Generated synthetic: $GENERATED_COUNT"
echo "  • Total: $((REAL_COUNT + GENERATED_COUNT))"
echo ""
echo "Negative samples: $EXISTING_NEGATIVE"
echo ""
echo "Next step: ./train_direct.sh \"$WAKE_WORD\""
echo ""
