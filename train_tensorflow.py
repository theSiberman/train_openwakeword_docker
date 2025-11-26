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

    # Extract features from audio clips using embed_clips (batch processing)
    print("🔊 Extracting audio features...")
    F = AudioFeatures()

    # Load all positive audio clips
    print("  → Loading positive samples...")
    positive_audio = []
    for i, clip_path in enumerate(positive_paths):
        try:
            sr, audio_data = wav.read(clip_path)

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1).astype(np.int16)

            # AudioFeatures expects int16 data
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype == np.int32:
                    audio_data = (audio_data / 65536).astype(np.int16)

            positive_audio.append(audio_data)
        except Exception as e:
            print(f"  ⚠ Skipping {clip_path}: {e}")

    # Load all negative audio clips
    print("  → Loading negative samples...")
    negative_audio = []
    for i, clip_path in enumerate(negative_paths):
        try:
            sr, audio_data = wav.read(clip_path)

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1).astype(np.int16)

            # AudioFeatures expects int16 data
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype == np.int32:
                    audio_data = (audio_data / 65536).astype(np.int16)

            negative_audio.append(audio_data)
        except Exception as e:
            print(f"  ⚠ Skipping {clip_path}: {e}")

    print(f"  → Loaded {len(positive_audio)} positive and {len(negative_audio)} negative clips")

    # Pad/truncate all clips to standard length (1.28 seconds at 16kHz = 20480 samples)
    # This matches openWakeWord's expected audio length
    target_length = 20480

    def pad_or_truncate(audio, target_len):
        """Pad or truncate audio to target length"""
        if len(audio) > target_len:
            return audio[:target_len]
        elif len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)), mode='constant')
        return audio

    print("  → Normalizing clip lengths...")
    positive_audio = np.array([pad_or_truncate(clip, target_length) for clip in positive_audio])
    negative_audio = np.array([pad_or_truncate(clip, target_length) for clip in negative_audio])

    # Use embed_clips for batch processing (returns shape: N, frames, 96)
    print("  → Computing embeddings for positive samples...")
    positive_features = F.embed_clips(positive_audio, batch_size=512, ncpu=4)
    print(f"    Positive features shape: {positive_features.shape}")

    print("  → Computing embeddings for negative samples...")
    negative_features = F.embed_clips(negative_audio, batch_size=512, ncpu=4)
    print(f"    Negative features shape: {negative_features.shape}")

    # Stack features and create labels
    X = np.vstack((positive_features, negative_features)).astype(np.float32)
    y = np.array([1]*len(positive_features) + [0]*len(negative_features), dtype=np.float32)

    print(f"📈 Training data shape: {X.shape}")
    print(f"   Positive samples: {np.sum(y == 1):.0f}")
    print(f"   Negative samples: {np.sum(y == 0):.0f}")
    print()

    # Build TensorFlow model - architecture from official openWakeWord notebook
    layer_dim = 32  # Official recommended size for pre-trained embeddings

    # X shape is (N, frames, 96) - need to flatten frames*96
    input_shape = (X.shape[1], X.shape[2])  # (frames, 96)
    flattened_dim = X.shape[1] * X.shape[2]  # frames * 96

    print(f"📐 Model architecture:")
    print(f"   Input shape: {input_shape} → flattened to {flattened_dim}")
    print(f"   Hidden layers: {layer_dim} units each")
    print()

    # Calculate class weights (15x weight for negative samples to reduce false positives)
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    total = len(y)
    weight_for_0 = (1 / negative_count) * (total / 2.0)
    weight_for_1 = (1 / positive_count) * (total / 2.0)
    class_weight = {0: weight_for_0 * 15, 1: weight_for_1}  # 15x weight (balanced to prevent overfitting)

    print(f"📊 Class weights:")
    print(f"   Negative class weight: {class_weight[0]:.4f}")
    print(f"   Positive class weight: {class_weight[1]:.4f}")
    print()

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Flatten(),  # Flatten (frames, 96) to (frames*96,)
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
        keras.layers.Dense(layer_dim),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("🧠 Training model...")
    print("   Using early stopping to prevent overfitting")
    print()

    class TrainingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                val_loss = logs.get('val_loss', 0)
                val_acc = logs.get('val_accuracy', 0)
                if val_loss > 0:
                    print(f"  → Epoch {epoch+1}/50 | Train Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
                else:
                    print(f"  → Epoch {epoch+1}/50 | Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.2%}")

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Stop if val_loss doesn't improve for 10 epochs
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        epochs=50,  # Reduced from 100, early stopping will cut off sooner
        batch_size=256,  # Smaller batches for more updates
        validation_split=0.15,  # Hold out 15% for validation
        class_weight=class_weight,  # 15x weight on negatives
        verbose=0,
        callbacks=[TrainingCallback(), early_stopping]
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
