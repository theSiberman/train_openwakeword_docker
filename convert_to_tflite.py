#!/usr/bin/env python3.10
"""Standalone script to convert ONNX to TFLite"""
import sys
import os
import shutil

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "hey_zardoz"

    import onnx2tf
    import tensorflow as tf

    print(f"📦 Converting {model_name}.onnx to TFLite...")

    onnx2tf.convert(
        input_onnx_file_path=f"{model_name}.onnx",
        output_folder_path=f"{model_name}_tf",
        copy_onnx_file=False,
        non_verbose=True
    )

    # Find SavedModel directory
    savedmodel_dir = None
    for item in os.listdir(f"{model_name}_tf"):
        if "savedmodel" in item.lower():
            savedmodel_dir = os.path.join(f"{model_name}_tf", item)
            break

    if not savedmodel_dir:
        savedmodel_dir = f"{model_name}_tf"

    print(f"  → Converting SavedModel to TFLite")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    tflite_model = converter.convert()

    with open(f"{model_name}.tflite", "wb") as f:
        f.write(tflite_model)

    file_size = os.path.getsize(f"{model_name}.tflite")
    print(f"  ✓ TFLite model saved ({file_size:,} bytes)")

    shutil.rmtree(f"{model_name}_tf", ignore_errors=True)

    print("\n🎉 TFLite conversion complete!")

if __name__ == "__main__":
    main()
