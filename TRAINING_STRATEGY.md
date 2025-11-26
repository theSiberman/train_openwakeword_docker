# Optimal Training Strategy for Your Wake Word

## Your Situation

- ✅ **441 real recordings** of the wake word (excellent!)
- ✅ Using **Alan Medium TTS** for Home Assistant responses
- ⚠️ **Problem**: Model triggers on TTS responses (feedback loop)

## Solution: 3-Phase Training Approach

### Phase 1: Organize Real Samples

```bash
# Put your 441 recordings in the real_samples directory
ls real_samples/*.wav | wc -l  # Should show 441

# Ensure they're 16kHz mono WAV files
# If needed, convert with:
# for f in real_samples/*.wav; do
#   ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.wav}_converted.wav"
# done
```

### Phase 2: Generate Training Data

With 441 real samples, generate **minimal synthetic** to reach ~1500-2000 total positives:

```bash
# Generate 1500-2000 synthetic positives (will use 3 voices, NOT Alan)
./generate_samples_direct.sh "your wake word" --positive-count 1500

# This will:
# - Use your 441 real samples
# - Generate 1059 synthetic (1500 - 441 = 1059)
# - Distribute across: Lessac (US Female), Alba (GB Female), Libritts (US Male)
# - EXCLUDES: Alan Medium (your TTS voice)
```

### Phase 3: Generate TTS Negative Samples

**Critical for preventing feedback loop:**

```bash
# Generate 500 samples of Alan saying common TTS responses
./generate_tts_negatives.sh 500

# This creates negative samples like:
# - "Okay"
# - "Turning on the lights"
# - "Done"
# - etc.
```

This teaches the model: **"Alan's voice = NOT the wake word"**

### Phase 4: Train the Model

```bash
# Train with the full dataset
python3.10 train_tensorflow.py "your wake word" model_name
```

**Expected data distribution:**
```
Positive samples: ~1941 total
  • Real recordings: 441 (primary quality)
  • Synthetic (3 voices): 1500

Negative samples: ~6500 total
  • Speech/noise: 6000 (from LibriSpeech)
  • TTS responses (Alan): 500 (prevents feedback)
```

---

## Why This Works

### 1. **Real Samples Are King** (441 recordings)
- Your voice, your microphone, your environment
- Model learns YOUR pronunciation and acoustics
- Dramatically improves accuracy

### 2. **Synthetic Samples Add Variety** (1500 samples)
- Different voices (NOT Alan) for generalization
- Acoustic variations (speed, pitch, noise)
- Helps model work for other users/accents

### 3. **TTS Negatives Prevent Feedback** (500 samples)
- Explicitly teaches: "Alan saying stuff ≠ wake word"
- Breaks the feedback loop
- Model learns to ignore TTS responses

### 4. **Negative/Positive Ratio** (6500:1941 ≈ 3.3:1)
- Higher negative ratio = fewer false positives
- 10x class weighting further emphasizes this
- Reduces random triggering

---

## Sample Count Recommendations

| Your Real Samples | Synthetic to Generate | TTS Negatives | Total Positives |
|-------------------|----------------------|---------------|-----------------|
| 441 (current)     | 1500-2000            | 500           | 1941-2441       |
| 200-300           | 2500-3000            | 500           | 2700-3300       |
| 100-150           | 3000-3500            | 500           | 3100-3650       |
| <100              | 4000-5000            | 500           | 4100-5100       |

**Rule of thumb**: More real samples = fewer synthetic needed

---

## Advanced: Fine-Tuning Detection Threshold

After training, if you still get occasional false positives:

### In Home Assistant (`configuration.yaml`):

```yaml
wyoming_openwakeword:
  models:
    - your_model
  threshold: 0.7  # Default is 0.5
  # Higher = fewer false positives, but might miss some real triggers
  # Lower = catches more wake words, but more false positives
```

**Recommended thresholds:**
- `0.5`: Default (balanced)
- `0.6`: Slightly stricter (good for noisy environments)
- `0.7`: Strict (use if you have feedback loops)
- `0.8`: Very strict (may miss some wake words)

### Testing Thresholds

```bash
# Test your model with different thresholds (on your dev machine)
from openwakeword import Model

model = Model(wakeword_models=["your_model.tflite"], inference_framework="tflite")

# Test with audio file
import numpy as np
import scipy.io.wavfile as wav

sr, audio = wav.read("test_sample.wav")
predictions = model.predict(audio)

# Try different thresholds
for threshold in [0.5, 0.6, 0.7]:
    if predictions['your_model'] > threshold:
        print(f"Triggered at threshold {threshold}")
```

---

## Complete Workflow (RunPod)

```bash
# 1. Ensure real samples are in place
ls real_samples/*.wav | wc -l  # Should show 441

# 2. Generate synthetic positives (excludes Alan)
./generate_samples_direct.sh "your wake word" --positive-count 1500

# 3. Generate TTS negatives (uses Alan)
./generate_tts_negatives.sh 500

# 4. Train the model
python3.10 train_tensorflow.py "your wake word" model_name

# 5. Deploy to Home Assistant
scp model_name.tflite homeassistant:/config/custom_components/openwakeword/models/
```

---

## Troubleshooting

### Still Getting Feedback Loops?

1. **Increase TTS negatives**: Try 1000+ samples
   ```bash
   ./generate_tts_negatives.sh 1000
   ```

2. **Raise detection threshold** in Home Assistant (0.6 → 0.7)

3. **Add your actual TTS responses as negatives**:
   - Record Home Assistant saying its common responses
   - Put in `negative_samples/` directory
   - Retrain

### Model Not Responding Enough?

1. **Check you have real samples**: Must be in `real_samples/`
2. **Lower threshold**: Try 0.4-0.5
3. **Record MORE real samples**: 500-1000+ is ideal

### Model Too Sensitive?

1. **More TTS negatives**: Generate 1000+
2. **Higher threshold**: 0.6-0.8
3. **More general negative samples**: Increase `--negative-count 10000`

---

## Expected Model Performance

With 441 real samples + proper training:

- ✅ **True positive rate**: 95-98% (catches your wake word)
- ✅ **False positive rate**: <0.5 per hour (very few false triggers)
- ✅ **TTS immunity**: Won't trigger on Alan's responses
- ✅ **Generalization**: Works with slight pronunciation variations

---

**Next Steps**: Run Phase 2-4 on RunPod with the commands above!
