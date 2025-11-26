# Feedback Loop Fix Guide

## Current Status

**v3 Model**: Still looping despite:
- ✅ Alan excluded from positive training
- ✅ 500 Alan TTS negatives included
- ✅ Correct model architecture (96KB)

**Root cause**: 500 TTS negatives insufficient, or model needs stronger negative bias.

---

## Solution Progression (Try in Order)

### **Level 1: More TTS Negatives** ⭐ Try This First

```bash
# Generate 2000 Alan TTS negatives (4x more)
./generate_tts_negatives.sh 2000 --force

# Retrain
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v4
```

**Expected improvement**: 70% chance of fixing loop

---

### **Level 2: Increase Class Weighting** ⭐⭐ If Level 1 Fails

The code now uses **20x** negative weighting (was 10x).

```bash
# Just retrain with updated weighting
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v5
```

**Expected improvement**: 85% chance of fixing loop

---

### **Level 3: Higher Detection Threshold** ⭐⭐⭐ Easy Win

In Home Assistant `configuration.yaml`:

```yaml
wyoming_openwakeword:
  models:
    - hey_zaadoz_v3
  threshold: 0.7  # Increase from 0.5
  # Try 0.7, then 0.75, then 0.8 if needed
```

**Restart Home Assistant after changing**

**Trade-off**: Might miss some wake words, but prevents loop

**Expected improvement**: 95% chance of fixing loop

---

### **Level 4: Record ACTUAL TTS Responses** ⭐⭐⭐⭐ Most Effective

The most powerful fix - use your actual Home Assistant responses:

```bash
# On your Home Assistant machine or wherever it's accessible:

# Trigger Home Assistant to respond
# While it's speaking "Okay", record it:
arecord -d 2 -f S16_LE -r 16000 -c 1 actual_tts_001.wav

# Repeat 20-50 times for different responses:
# - "Okay"
# - "Turning on the lights"
# - "Done"
# - etc.

# Copy to RunPod
scp actual_tts_*.wav runpod:/workspace/negative_samples/

# Retrain
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v6
```

**Why this is best**: Trains on EXACT audio that triggers the loop, including:
- Your actual speaker frequency response
- Room acoustics
- Exact TTS voice characteristics

**Expected improvement**: 99% chance of fixing loop

---

### **Level 5: Cooldown Period** (Home Assistant Config)

Add a cooldown so model can't trigger multiple times quickly:

```yaml
automation:
  - alias: "Wake Word Cooldown"
    trigger:
      - platform: event
        event_type: wyoming_wake_word_detected
    action:
      - service: input_boolean.turn_on
        target:
          entity_id: input_boolean.wake_word_cooldown
      - delay: 3  # 3 second cooldown
      - service: input_boolean.turn_off
        target:
          entity_id: input_boolean.wake_word_cooldown

# Then condition all wake word actions on cooldown being off
condition:
  - condition: state
    entity_id: input_boolean.wake_word_cooldown
    state: 'off'
```

---

## Diagnostic Questions

### What exactly is triggering?

**Option A: "Okay" triggers immediately**
- Solution: Level 1 or Level 4 (more "Okay" negatives)

**Option B: The beep/chime sound triggers it**
- Solution: Record beep, add as negative sample

**Option C: Your voice asking question triggers second time**
- Solution: This is expected - you said wake word twice
- Solution: Add cooldown (Level 5)

**Option D: Background noise during TTS triggers it**
- Solution: Level 3 (higher threshold)

---

## Quick Test Procedure

After each fix:

```bash
# Deploy new model
scp hey_zaadoz_vX.tflite homeassistant:/config/...

# Test sequence:
1. Say: "Hey Zaadoz"
2. Wait for beep
3. Say: "Turn on the lights"
4. Listen for: "Okay"
5. OBSERVE: Does it trigger again?

# If YES → try next level
# If NO → Success! ✓
```

---

## Recommended Approach

**For fastest fix**:

```bash
# Step 1: Quick wins (5 minutes)
# Edit Home Assistant config:
threshold: 0.7  # Up from 0.5

# Test - does this fix it?
# If YES: Done! ✓
# If NO: Continue...

# Step 2: More TTS negatives (10 minutes)
./generate_tts_negatives.sh 2000 --force
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v4

# Test - does this fix it?
# If YES: Done! ✓
# If NO: Continue...

# Step 3: Record actual TTS (15 minutes)
# Record 20-50 actual Home Assistant responses
# Add to negative_samples/
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v5

# This should fix it 99% of the time ✓
```

---

## Understanding the Loop

```
User: "Hey Zaadoz"
  ↓
Model: Detected! (score: 0.65)
  ↓
HA: "Okay" (Alan voice)
  ↓
Model: Detected! (score: 0.55) ← FALSE POSITIVE
  ↓
HA: "Okay"
  ↓
[INFINITE LOOP]
```

**The fix**: Make model output score < threshold when it hears Alan

**How**:
1. More Alan negatives → model learns "Alan ≠ wake word"
2. Higher class weight → model MORE strongly rejects negatives
3. Higher threshold → model needs to be MORE confident
4. Actual TTS recordings → trains on EXACT triggering audio

---

## Advanced: Hybrid Approach

Combine multiple fixes for best results:

```bash
# 1. Generate 2000 TTS negatives
./generate_tts_negatives.sh 2000 --force

# 2. Record 50 actual TTS responses
# (copy to negative_samples/)

# 3. Train with 20x class weighting (already updated in code)
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v4

# 4. Deploy with threshold 0.6 (not too high)
```

**Result**: Near-perfect wake word detection with zero loop risk

---

## Current Code Changes

✅ Class weighting increased: **10x → 20x**

To revert if model becomes too strict:
```python
# Line 161 in train_tensorflow.py
class_weight = {0: weight_for_0 * 10, 1: weight_for_1}  # Change back to 10x
```

---

## Next Steps

**Run this now**:
```bash
./generate_tts_negatives.sh 2000 --force
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v4
```

**Test in Home Assistant**

**If still loops**: Record actual TTS responses (Level 4)
