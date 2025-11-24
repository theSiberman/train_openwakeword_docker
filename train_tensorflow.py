#!/usr/bin/env python3.10
"""Train wake word model using TensorFlow and export directly to TFLite"""
import glob
import os
import numpy as np
from openwakeword.utils import AudioFeatures
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow import keras

def main():
    OUTPUT_MODEL = "hey_zardoz"

    clip_paths = sorted(glob.glob(os.path.join("generated_samples", "*.wav")))

    if len(clip_paths) == 0:
        print("❌ Error: No audio samples found!")
        return 1

    print(f"\n🧠 Training Custom Model with {len(clip_paths)} clips...")

    # Extract features from audio clips
    print("  → Extracting audio features...")
    F = AudioFeatures()
    all_features = []

    for i, clip_path in enumerate(clip_paths):
        if (i+1) % 100 == 0:
            print(f"  → Processed {i+1}/{len(clip_paths)} clips")

        try:
            sr, audio_data = wav.read(clip_path)
            if audio_data.dtype != np.int16:
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
    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.ones((X.shape[0], 1), dtype=np.float32)
    print(f"  → Training data shape: {X.shape}")

    # Build TensorFlow model
    layer_dim = 32
    input_dim = X.shape[1]

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("\n  → Training model...")
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                    print(f"  → Epoch {epoch+1}/50, Loss: {logs['loss']:.4f}")
                    if (epoch+1) % 10 == 0 else None
            )
        ]
    )

    # Export as SavedModel first
    print("\n📦 Exporting TensorFlow SavedModel...")
    model.export(f"{OUTPUT_MODEL}_savedmodel")
    print("  ✓ SavedModel exported")

    # Convert to TFLite
    print("\n📦 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{OUTPUT_MODEL}_savedmodel")
    tflite_model = converter.convert()

    with open(f"{OUTPUT_MODEL}.tflite", "wb") as f:
        f.write(tflite_model)

    file_size = os.path.getsize(f"{OUTPUT_MODEL}.tflite")
    print(f"  ✓ TFLite model saved ({file_size:,} bytes)")

    # Also export to ONNX for compatibility
    print("\n📦 Exporting to ONNX...")
    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)
        output_path = f"{OUTPUT_MODEL}.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=18, output_path=output_path)
        print("  ✓ ONNX model saved")
    except Exception as e:
        print(f"  ⚠ ONNX export failed: {e}")
        print("  → TFLite model is the primary output")

    # Cleanup
    import shutil
    shutil.rmtree(f"{OUTPUT_MODEL}_savedmodel", ignore_errors=True)

    print("\n🎉 Training complete!")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_MODEL}.tflite (TensorFlow Lite)")
    if os.path.exists(f"{OUTPUT_MODEL}.onnx"):
        print(f"  - {OUTPUT_MODEL}.onnx (ONNX)")

    return 0

if __name__ == "__main__":
    exit(main())
