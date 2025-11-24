#!/bin/bash
# Direct training without Docker - for RunPod or environments where Docker isn't available
set -e

# --- Configuration ---
WAKE_WORD="hey zaadoz"
OUTPUT_MODEL="hey_zardoz"
VOICE_DIR="./models"

# Parse command-line arguments
KEEP_SAMPLES=false
for arg in "$@"; do
    case $arg in
        --keep-samples)
            KEEP_SAMPLES=true
            shift
            ;;
    esac
done

echo "========================================"
echo "OpenWakeWord Training (Direct)"
echo "========================================"
echo "Wake word: $WAKE_WORD"
echo "Model name: $OUTPUT_MODEL"
echo "Keep samples: $KEEP_SAMPLES"
echo ""

# Step 1: Generate synthetic samples (if needed)
SAMPLE_COUNT=$(ls generated_samples/*.wav 2>/dev/null | wc -l)
TARGET_SAMPLES=2000

if [ "$SAMPLE_COUNT" -ge "$TARGET_SAMPLES" ]; then
    echo "🔊 Using existing $SAMPLE_COUNT audio clips..."
else
    echo "🔊 Generating synthetic audio clips (target: $TARGET_SAMPLES)..."
    mkdir -p generated_samples

    # Start generation in background
    python3.10 /opt/piper-sample-generator/generate_samples.py "$WAKE_WORD" \
        --model $VOICE_DIR/en_US-lessac-medium.onnx \
        --max-samples $TARGET_SAMPLES \
        --output-dir generated_samples &

    GENERATION_PID=$!

    # Monitor progress
    while kill -0 $GENERATION_PID 2>/dev/null; do
        CURRENT=$(ls generated_samples/*.wav 2>/dev/null | wc -l)
        echo "  → Generated $CURRENT/$TARGET_SAMPLES samples..."
        sleep 5
    done

    # Wait for completion
    wait $GENERATION_PID
    FINAL_COUNT=$(ls generated_samples/*.wav 2>/dev/null | wc -l)
    echo "  ✓ Generation complete: $FINAL_COUNT samples"
fi

# Step 2: Train the model
python3.10 << 'PYTHON_SCRIPT'
import glob, os, shutil
import numpy as np
from openwakeword.utils import AudioFeatures
import torch
from torch import nn
import scipy.io.wavfile as wav

OUTPUT_MODEL_ENV = os.environ.get("OUTPUT_MODEL")
KEEP_SAMPLES_ENV = os.environ.get("KEEP_SAMPLES", "false").lower() == "true"

clip_paths = sorted(glob.glob(os.path.join("generated_samples", "*.wav")))

if len(clip_paths) == 0:
    print("❌ Error: No audio samples were generated!")
    exit(1)

print(f"\n🧠 Training Custom Model with {len(clip_paths)} clips...")

# Extract features from audio clips
print("  → Extracting audio features...")
F = AudioFeatures()
all_features = []

for i, clip_path in enumerate(clip_paths):
    if (i+1) % 250 == 0:
        print(f"  → Processed {i+1}/{len(clip_paths)} clips")

    try:
        sr, audio_data = wav.read(clip_path)
        # AudioFeatures expects int16 data, do not normalize to float32
        if audio_data.dtype != np.int16:
            # Convert float32 back to int16 if needed
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype == np.int32:
                audio_data = (audio_data / 65536).astype(np.int16)

        features = F._get_embeddings(audio_data)
        if features.shape[0] > 0:
            all_features.append(features)
    except Exception as e:
        print(f"  ⚠ Skipping {clip_path}: {e}")

print(f"  ✓ Extracted features from {len(all_features)} clips")

# Prepare training data
X = np.concatenate(all_features, axis=0)
y = np.ones((X.shape[0], 1))
print(f"  → Training data shape: {X.shape}")

# Build model
layer_dim = 32
input_dim = X.shape[1]  # Features are already flat (samples, features)
model = nn.Sequential(
    nn.Linear(input_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, 1),
    nn.Sigmoid(),
)

# Train
print("\n  → Training model...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

for epoch in range(50):
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = torch.nn.functional.binary_cross_entropy(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"  → Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

# Export to ONNX
print("\n📦 Exporting to ONNX...")
model.eval()
dummy_input = torch.randn(1, input_dim)
output_path = f"{OUTPUT_MODEL_ENV}.onnx"
torch.onnx.export(
    model, dummy_input, output_path,
    export_params=True, opset_version=18
)

print("  ✓ ONNX model saved")

# Convert to TFLite
print("📦 Converting to TFLite...")
try:
    import onnx
    import tensorflow as tf
    import subprocess
    import sys

    # Try using onnx2tf (better ONNX opset 18 support)
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=f"{OUTPUT_MODEL_ENV}.onnx",
            output_folder_path=f"{OUTPUT_MODEL_ENV}_tf",
            copy_onnx_file=False,
            non_verbose=True
        )

        # Find the saved model directory
        import os
        savedmodel_dir = None
        for item in os.listdir(f"{OUTPUT_MODEL_ENV}_tf"):
            if "savedmodel" in item.lower():
                savedmodel_dir = os.path.join(f"{OUTPUT_MODEL_ENV}_tf", item)
                break

        if not savedmodel_dir:
            savedmodel_dir = f"{OUTPUT_MODEL_ENV}_tf"

        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
        tflite_model = converter.convert()

        with open(f"{OUTPUT_MODEL_ENV}.tflite", "wb") as f:
            f.write(tflite_model)

        print("  ✓ TFLite model saved (via onnx2tf)")
        shutil.rmtree(f"{OUTPUT_MODEL_ENV}_tf", ignore_errors=True)
    except ImportError:
        print("  → onnx2tf not available, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx2tf"])
        import onnx2tf

        onnx2tf.convert(
            input_onnx_file_path=f"{OUTPUT_MODEL_ENV}.onnx",
            output_folder_path=f"{OUTPUT_MODEL_ENV}_tf",
            copy_onnx_file=False,
            non_verbose=True
        )

        import os
        savedmodel_dir = None
        for item in os.listdir(f"{OUTPUT_MODEL_ENV}_tf"):
            if "savedmodel" in item.lower():
                savedmodel_dir = os.path.join(f"{OUTPUT_MODEL_ENV}_tf", item)
                break

        if not savedmodel_dir:
            savedmodel_dir = f"{OUTPUT_MODEL_ENV}_tf"

        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
        tflite_model = converter.convert()

        with open(f"{OUTPUT_MODEL_ENV}.tflite", "wb") as f:
            f.write(tflite_model)

        print("  ✓ TFLite model saved (via onnx2tf)")
        shutil.rmtree(f"{OUTPUT_MODEL_ENV}_tf", ignore_errors=True)
except Exception as e:
    print(f"  ⚠ TFLite conversion failed: {e}")
    print("  → ONNX model is still available")

# Cleanup
if not KEEP_SAMPLES_ENV:
    print("\n🧹 Cleaning up generated samples...")
    shutil.rmtree("generated_samples", ignore_errors=True)
else:
    print(f"\n💾 Generated samples preserved in ./generated_samples/ ({len(clip_paths)} files)")

print("\n🎉 Training complete!")
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Output files in current directory:"
ls -lh "${OUTPUT_MODEL}.tflite" 2>/dev/null || echo "  ⚠ Warning: Expected output file not found"
echo ""
echo "You can now use this model with OpenWakeWord"
echo "To train again with different parameters, just run ./train_direct.sh"
if [ "$KEEP_SAMPLES" = true ]; then
    echo ""
    echo "Generated samples are in ./generated_samples/"
fi
echo ""
