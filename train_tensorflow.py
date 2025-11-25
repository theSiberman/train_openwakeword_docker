#!/usr/bin/env python3.10
"""Train wake word model using TensorFlow and export directly to TFLite"""
import glob
import os
import sys
import numpy as np

# Force CPU usage (workaround for RTX 5090 GPU compatibility issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from openwakeword.utils import AudioFeatures
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow import keras

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: train_tensorflow.py <wake_word> [output_model_name]")
        print("Example: train_tensorflow.py \"hey zaadoz\"")
        return 1

    wake_word = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else wake_word.replace(' ', '_').lower()

    # Directories
    generated_dir = "./generated_samples"
    real_dir = "./real_samples"
    negative_dir = "./negative_samples"

    print("=" * 40)
    print("OpenWakeWord Training (TensorFlow)")
    print("=" * 40)
    print(f"Wake word: {wake_word}")
    print(f"Output model: {output_model}")
    print()

    # Collect all positive samples (generated + real)
    positive_paths = sorted(glob.glob(os.path.join(generated_dir, "*.wav")))
    positive_paths += sorted(glob.glob(os.path.join(real_dir, "*.wav")))

    # Collect all negative samples
    negative_paths = sorted(glob.glob(os.path.join(negative_dir, "*.wav")))

    print(f"📊 Training data:")
    print(f"  • Positive samples: {len(positive_paths)}")
    print(f"  • Negative samples: {len(negative_paths)}")
    print()

    if len(positive_paths) == 0:
        print("❌ Error: No positive samples found!")
        print("   Run: ./generate_samples_direct.sh \"" + wake_word + "\" first")
        return 1

    if len(negative_paths) == 0:
        print("⚠ Warning: No negative samples found. Training without negative samples will be less robust.")
        print("   Consider running: ./generate_samples_direct.sh \"" + wake_word + "\"")

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

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # AudioFeatures expects int16 data
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype == np.int32:
                    audio_data = (audio_data / 65536).astype(np.int16)

            features = F._get_embeddings(audio_data)

            # Ensure features are 2D and have valid shape
            if len(features.shape) == 1:
                # Skip 1D features (likely too short or corrupted)
                continue
            if features.shape[0] > 0 and len(features.shape) == 2:
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

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # AudioFeatures expects int16 data
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype == np.int32:
                    audio_data = (audio_data / 65536).astype(np.int16)

            features = F._get_embeddings(audio_data)

            # Ensure features are 2D and have valid shape
            if len(features.shape) == 1:
                # Skip 1D features (likely too short or corrupted)
                continue
            if features.shape[0] > 0 and len(features.shape) == 2:
                all_features.append(features)
                all_labels.append(np.zeros((features.shape[0], 1)))  # Label 0 for negative
        except Exception as e:
            print(f"  ⚠ Skipping {clip_path}: {e}")

    print(f"  ✓ Extracted features from {len(all_features)} clips total")
    print()

    # Prepare training data
    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.float32)

    print(f"📈 Training data shape: {X.shape}")
    print(f"   Positive samples: {np.sum(y == 1):.0f}")
    print(f"   Negative samples: {np.sum(y == 0):.0f}")
    print()

    # Build TensorFlow model
    layer_dim = 64  # Increased from 32 for better capacity with more data
    input_dim = X.shape[1]

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("🧠 Training model...")

    class TrainingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f"  → Epoch {epoch+1}/100, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.2%}")

    history = model.fit(
        X, y,
        epochs=100,  # Increased from 50 for better convergence
        batch_size=32,
        verbose=0,
        callbacks=[TrainingCallback()]
    )

    print()

    # Export as SavedModel first
    print("📦 Exporting TensorFlow SavedModel...")
    model.export(f"{output_model}_savedmodel")
    print("  ✓ SavedModel exported")

    # Convert to TFLite
    print()
    print("📦 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{output_model}_savedmodel")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = f"{output_model}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"  ✓ TFLite model saved: {tflite_path}")
    print(f"  → TFLite size: {tflite_size:.1f} KB")

    # Cleanup
    import shutil
    shutil.rmtree(f"{output_model}_savedmodel", ignore_errors=True)

    print()
    print("🎉 Training complete!")
    print()
    print("=" * 40)
    print("Training Complete!")
    print("=" * 40)
    print()
    print("Output files:")
    print(f"  - {tflite_path} (TensorFlow Lite - {tflite_size:.1f} KB)")
    print()
    print("For Home Assistant, copy the .tflite file to your config directory:")
    print(f"  custom_components/openwakeword/models/{output_model}.tflite")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
