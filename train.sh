#!/bin/bash
# Training script for OpenWakeWord models
# Run this after setup.sh to train custom wake word models

set -e

# --- Configuration ---
WAKE_WORD="hey zaadoz"
OUTPUT_MODEL="hey_zardoz"
VOICE_DIR="./models"
CUSTOM_IMAGE="oww-training-env"

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
echo "OpenWakeWord Training"
echo "========================================"
echo "Wake word: $WAKE_WORD"
echo "Model name: $OUTPUT_MODEL"
echo "Keep samples: $KEEP_SAMPLES"
echo ""

# Check if custom image exists
if ! docker image inspect $CUSTOM_IMAGE >/dev/null 2>&1; then
    echo "❌ Error: Custom image '$CUSTOM_IMAGE' not found"
    echo "Please run ./setup.sh first to create the training environment"
    exit 1
fi

# Run training in Docker container
echo "Starting training process..."
echo ""

docker run --rm \
    --entrypoint /bin/bash \
    -v "$(pwd)":/app \
    -v "$(pwd)/$VOICE_DIR":/app/models \
    -e TARGET_PHRASE="$WAKE_WORD" \
    -e OUTPUT_MODEL="$OUTPUT_MODEL" \
    -e KEEP_SAMPLES="$KEEP_SAMPLES" \
    $CUSTOM_IMAGE \
    -c '
        set -e

        # Step 1: Generate synthetic samples (if needed)
        SAMPLE_COUNT=$(ls /app/generated_samples/*.wav 2>/dev/null | wc -l)
        if [ "$SAMPLE_COUNT" -ge 2000 ]; then
            echo "🔊 Using existing $SAMPLE_COUNT audio clips..."
        else
            echo "🔊 Generating synthetic audio clips..."
            mkdir -p /app/generated_samples
            python3 /opt/piper-sample-generator/generate_samples.py "$TARGET_PHRASE" \
                --model /app/models/en_US-lessac-medium.onnx \
                --max-samples 2000 \
                --output-dir /app/generated_samples
        fi

        # Step 2: Train the model
        python3 << 'PYTHON_SCRIPT'
import glob, os, shutil
import numpy as np
from openwakeword.utils import AudioFeatures
import torch
from torch import nn
import scipy.io.wavfile as wav

OUTPUT_MODEL_ENV = os.environ.get("OUTPUT_MODEL")
KEEP_SAMPLES_ENV = os.environ.get("KEEP_SAMPLES", "false").lower() == "true"

clip_paths = sorted(glob.glob(os.path.join("/app/generated_samples", "*.wav")))

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
output_path = f"/app/{OUTPUT_MODEL_ENV}.onnx"
torch.onnx.export(
    model, dummy_input, output_path,
    export_params=True, opset_version=12
)

print("  ✓ ONNX model saved")

# Convert to TFLite
print("📦 Converting to TFLite...")
try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(f"/app/{OUTPUT_MODEL_ENV}.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f"/app/{OUTPUT_MODEL_ENV}_tf")

    converter = tf.lite.TFLiteConverter.from_saved_model(f"/app/{OUTPUT_MODEL_ENV}_tf")
    tflite_model = converter.convert()

    with open(f"/app/{OUTPUT_MODEL_ENV}.tflite", "wb") as f:
        f.write(tflite_model)

    print("  ✓ TFLite model saved")
    shutil.rmtree(f"/app/{OUTPUT_MODEL_ENV}_tf", ignore_errors=True)
except Exception as e:
    print(f"  ⚠ TFLite conversion failed: {e}")
    print("  → ONNX model is still available")

# Cleanup
if not KEEP_SAMPLES_ENV:
    print("\n🧹 Cleaning up generated samples...")
    shutil.rmtree("/app/generated_samples", ignore_errors=True)
else:
    print(f"\n💾 Generated samples preserved in ./generated_samples/ ({len(clip_paths)} files)")

print("\n🎉 Training complete!")
PYTHON_SCRIPT
    '

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Output files in current directory:"
ls -lh "${OUTPUT_MODEL}.tflite" 2>/dev/null || echo "  ⚠ Warning: Expected output file not found"
echo ""
echo "You can now use this model with OpenWakeWord"
echo "To train again with different parameters, just run ./train.sh"
if [ "$KEEP_SAMPLES" = true ]; then
    echo ""
    echo "Generated samples are in ./generated_samples/"
fi
echo ""
