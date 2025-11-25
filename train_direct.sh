#!/bin/bash
# Train OpenWakeWord model from generated samples
# Usage: ./train_direct.sh "wake word phrase" [output_model_name]

set -e

# --- Configuration ---
WAKE_WORD="${1:-hey zaadoz}"
OUTPUT_MODEL="${2:-$(echo "$WAKE_WORD" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')}"
GENERATED_DIR="./generated_samples"
REAL_DIR="./real_samples"
NEGATIVE_DIR="./negative_samples"

# Validate inputs
if [ -z "$WAKE_WORD" ]; then
    echo "❌ Error: Wake word phrase is required"
    echo "Usage: $0 \"wake word phrase\" [output_model_name]"
    exit 1
fi

echo "========================================"
echo "OpenWakeWord Training"
echo "========================================"
echo "Wake word: $WAKE_WORD"
echo "Output model: $OUTPUT_MODEL"
echo ""

# Determine Python command
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

# --- Step 1: Verify samples exist ---
POSITIVE_COUNT=$(ls "$GENERATED_DIR"/*.wav "$REAL_DIR"/*.wav 2>/dev/null | wc -l)
NEGATIVE_COUNT=$(ls "$NEGATIVE_DIR"/*.wav 2>/dev/null | wc -l)

echo "Sample counts:"
echo "  • Positive (wake word): $POSITIVE_COUNT"
echo "  • Negative (background): $NEGATIVE_COUNT"
echo ""

if [ $POSITIVE_COUNT -eq 0 ]; then
    echo "❌ Error: No positive samples found!"
    echo "   Run: ./generate_samples_direct.sh \"$WAKE_WORD\" first"
    exit 1
fi

if [ $NEGATIVE_COUNT -eq 0 ]; then
    echo "⚠ Warning: No negative samples found. Training without negative samples will be less robust."
    echo "   Consider running: ./generate_samples_direct.sh \"$WAKE_WORD\""
fi

# --- Step 2: Train the model with Python ---
echo "Training model..."
echo ""

$PYTHON_CMD << 'PYTHON_TRAIN'
import glob, os, shutil
import numpy as np
from openwakeword.utils import AudioFeatures
import torch
from torch import nn
import scipy.io.wavfile as wav

# Get configuration from environment
OUTPUT_MODEL = os.environ.get("OUTPUT_MODEL")
GENERATED_DIR = os.environ.get("GENERATED_DIR", "./generated_samples")
REAL_DIR = os.environ.get("REAL_DIR", "./real_samples")
NEGATIVE_DIR = os.environ.get("NEGATIVE_DIR", "./negative_samples")

# Collect all positive samples (generated + real)
positive_paths = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.wav")))
positive_paths += sorted(glob.glob(os.path.join(REAL_DIR, "*.wav")))

# Collect all negative samples
negative_paths = sorted(glob.glob(os.path.join(NEGATIVE_DIR, "*.wav")))

print(f"📊 Training data:")
print(f"  • Positive samples: {len(positive_paths)}")
print(f"  • Negative samples: {len(negative_paths)}")
print()

# Extract features from audio clips
print("🔊 Extracting audio features...")
F = AudioFeatures()
all_features = []
all_labels = []

# Process positive samples
for i, clip_path in enumerate(positive_paths):
    if (i+1) % 500 == 0:
        print(f"  → Processed {i+1}/{len(positive_paths)} positive clips")

    try:
        sr, audio_data = wav.read(clip_path)

        # AudioFeatures expects int16 data
        if audio_data.dtype != np.int16:
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype == np.int32:
                audio_data = (audio_data / 65536).astype(np.int16)

        features = F._get_embeddings(audio_data)
        if features.shape[0] > 0:
            all_features.append(features)
            all_labels.append(np.ones((features.shape[0], 1)))  # Label 1 for positive
    except Exception as e:
        print(f"  ⚠ Skipping {clip_path}: {e}")

# Process negative samples
for i, clip_path in enumerate(negative_paths):
    if (i+1) % 500 == 0:
        print(f"  → Processed {i+1}/{len(negative_paths)} negative clips")

    try:
        sr, audio_data = wav.read(clip_path)

        # AudioFeatures expects int16 data
        if audio_data.dtype != np.int16:
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype == np.int32:
                audio_data = (audio_data / 65536).astype(np.int16)

        features = F._get_embeddings(audio_data)
        if features.shape[0] > 0:
            all_features.append(features)
            all_labels.append(np.zeros((features.shape[0], 1)))  # Label 0 for negative
    except Exception as e:
        print(f"  ⚠ Skipping {clip_path}: {e}")

print(f"  ✓ Extracted features from {len(all_features)} clips total")
print()

# Prepare training data
X = np.concatenate(all_features, axis=0)
y = np.concatenate(all_labels, axis=0)

print(f"📈 Training data shape: {X.shape}")
print(f"   Positive samples: {np.sum(y == 1)}")
print(f"   Negative samples: {np.sum(y == 0)}")
print()

# Build model
layer_dim = 64  # Increased from 32 for better capacity with more data
input_dim = X.shape[1]

model = nn.Sequential(
    nn.Linear(input_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(layer_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(layer_dim, 1),
    nn.Sigmoid(),
)

# Train
print("🧠 Training model...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

epochs = 100  # Increased from 50 for better convergence
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = torch.nn.functional.binary_cross_entropy(predictions, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        # Calculate accuracy
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_tensor).float().mean()
        print(f"  → Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2%}")

print()

# Export to ONNX
print("📦 Exporting to ONNX...")
model.eval()
dummy_input = torch.randn(1, input_dim)
output_path = f"{OUTPUT_MODEL}.onnx"
torch.onnx.export(
    model, dummy_input, output_path,
    export_params=True, opset_version=12
)
print(f"  ✓ ONNX model saved: {output_path}")

# Convert to TFLite
print()
print("📦 Converting to TFLite...")
try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(f"{OUTPUT_MODEL}.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f"{OUTPUT_MODEL}_tf")

    converter = tf.lite.TFLiteConverter.from_saved_model(f"{OUTPUT_MODEL}_tf")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = f"{OUTPUT_MODEL}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"  ✓ TFLite model saved: {tflite_path}")

    # Get file sizes
    onnx_size = os.path.getsize(output_path) / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"  → ONNX size: {onnx_size:.1f} KB")
    print(f"  → TFLite size: {tflite_size:.1f} KB")

    # Cleanup temp files
    shutil.rmtree(f"{OUTPUT_MODEL}_tf", ignore_errors=True)
except Exception as e:
    print(f"  ⚠ TFLite conversion failed: {e}")
    print(f"  → ONNX model is still available at {output_path}")

print()
print("🎉 Training complete!")
PYTHON_TRAIN

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Output files:"
ls -lh "${OUTPUT_MODEL}.onnx" "${OUTPUT_MODEL}.tflite" 2>/dev/null || ls -lh "${OUTPUT_MODEL}.onnx"
echo ""
echo "For Home Assistant, copy the .tflite file to your config directory:"
echo "  custom_components/openwakeword/models/${OUTPUT_MODEL}.tflite"
echo ""
