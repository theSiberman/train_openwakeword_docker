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

        # Generate to temporary directory
        TEMP_DIR="$GENERATED_DIR/${VOICE_NAME}_temp"
        mkdir -p "$TEMP_DIR"

        $PYTHON_CMD /opt/piper-sample-generator/generate_samples.py "$WAKE_WORD" \
            --model "$VOICE" \
            --max-samples $COUNT \
            --output-dir "$TEMP_DIR" \
            --slerp-weights 0.3 \
            --length-scales 0.8 1.0 1.2 \
            --noise-scales 0.4 0.667 0.8 2>&1 | grep -v "WARNING"

        # Rename files with voice prefix and move to main directory
        for file in "$TEMP_DIR"/*.wav; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                mv "$file" "$GENERATED_DIR/${VOICE_NAME}_${filename}"
            fi
        done
        rmdir "$TEMP_DIR"
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

# Calculate distribution: 70% speech, 30% noise (synthetic)
speech_count = int(target_count * 0.7)  # 4200
noise_count = target_count - speech_count  # 1800

count = 0

# 1. Download speech samples from LibriSpeech (well-supported, standard format)
try:
    print(f"  → Downloading {speech_count} speech clips from LibriSpeech...")
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    for i, sample in enumerate(dataset):
        if i >= speech_count:
            break
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))

        # Ensure audio is not too short (at least 0.5 seconds)
        if len(audio) < 8000:
            continue

        output_path = os.path.join(negative_dir, f"speech_{i:05d}.wav")
        sf.write(output_path, audio, 16000)
        count += 1
        if (i + 1) % 500 == 0:
            print(f"    Downloaded {i + 1}/{speech_count} speech clips...")
except Exception as e:
    print(f"  ⚠ Error downloading speech: {e}")
    print(f"     Trying alternative dataset...")
    # Try People's Speech as backup
    try:
        dataset = load_dataset("MLCommons/peoples_speech", split="train", streaming=True)
        for i, sample in enumerate(dataset):
            if count >= speech_count:
                break
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]

            if sr != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))

            if len(audio) < 8000:
                continue

            output_path = os.path.join(negative_dir, f"speech_{i:05d}.wav")
            sf.write(output_path, audio, 16000)
            count += 1
            if (i + 1) % 500 == 0:
                print(f"    Downloaded {i + 1}/{speech_count} speech clips...")
    except Exception as e2:
        print(f"  ⚠ Alternative dataset also failed: {e2}")

# 2. Generate synthetic noise samples
try:
    print(f"  → Generating {noise_count} synthetic noise samples...")
    np.random.seed(42)
    for i in range(noise_count):
        # Generate 2-4 second noise clips
        duration = np.random.uniform(2.0, 4.0)
        num_samples = int(16000 * duration)

        # Mix different noise types
        noise_type = i % 4
        if noise_type == 0:
            # White noise
            audio = np.random.randn(num_samples) * 0.1
        elif noise_type == 1:
            # Pink noise (1/f noise)
            white = np.random.randn(num_samples)
            from scipy import signal
            b, a = signal.butter(1, 0.1)
            audio = signal.filtfilt(b, a, white) * 0.15
        elif noise_type == 2:
            # Brown noise
            white = np.random.randn(num_samples)
            audio = np.cumsum(white) * 0.01
            audio = audio - np.mean(audio)
        else:
            # Synthetic room noise (multiple frequencies)
            t = np.linspace(0, duration, num_samples)
            audio = (
                0.05 * np.sin(2 * np.pi * 60 * t) +  # 60Hz hum
                0.03 * np.sin(2 * np.pi * 120 * t) +  # 120Hz harmonic
                0.02 * np.random.randn(num_samples)    # Background noise
            )

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.3

        output_path = os.path.join(negative_dir, f"noise_{i:05d}.wav")
        sf.write(output_path, audio.astype(np.float32), 16000)
        count += 1
        if (i + 1) % 500 == 0:
            print(f"    Generated {i + 1}/{noise_count} noise clips...")
except Exception as e:
    print(f"  ⚠ Error generating noise: {e}")

print(f"\n  ✓ Downloaded/generated {count} negative samples total")
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
