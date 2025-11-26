import os
import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- CONFIGURATION ---
# The folder containing your original recordings
INPUT_FOLDER = "./recorded_trigger_samples"

# The folder where the split clips will be saved
OUTPUT_FOLDER = "./split_samples"

# CHANGED: Set to m4a for your files
FILE_EXTENSION = ".m4a"

# --- ADAPTIVE SETTINGS ---
# Instead of a fixed number, we will calculate the threshold relative to the file.
# THRESHOLD_BUFFER: How much quieter than the average volume must silence be?
# 16dB is a good standard. If it splits too much, increase this (e.g. 20).
THRESHOLD_BUFFER = 16 

MIN_SILENCE_LEN = 300
KEEP_SILENCE = 100 
# ---------------------

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_audio():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        return

    ensure_dir(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(FILE_EXTENSION)]
    
    if not files:
        print(f"No {FILE_EXTENSION} files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"\nProcessing: {filename}...")

        try:
            audio = AudioSegment.from_file(file_path)
            
            # --- ADAPTIVE THRESHOLD CALCULATION ---
            # We take the average loudness (dBFS) and subtract the buffer.
            # Example: If file is -46dB, threshold becomes -62dB.
            adaptive_threshold = audio.dBFS - THRESHOLD_BUFFER

            print(f"  -> Average Volume: {audio.dBFS:.2f} dBFS")
            print(f"  -> Adaptive Threshold: {adaptive_threshold:.2f} dBFS")
            
            # Convert to Mono (Left Channel)
            left_channel_audio = audio.split_to_mono()[0]

            # Split based on silence using the ADAPTIVE threshold
            chunks = split_on_silence(
                left_channel_audio,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=adaptive_threshold,
                keep_silence=KEEP_SILENCE
            )

            if not chunks:
                print("  -> Result: No audio chunks found. (Try increasing THRESHOLD_BUFFER)")
                continue

            base_name = os.path.splitext(filename)[0]
            
            for i, chunk in enumerate(chunks):
                # Normalize chunk (optional)
                chunk = chunk.normalize()

                # Save as WAV (standard for clips)
                out_name = f"{base_name}_{i+1:03d}.wav"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)

                print(f"  -> Exporting: {out_name}")
                chunk.export(out_path, format="wav")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print("\nDone! Check the output folder.")

if __name__ == "__main__":
    process_audio()