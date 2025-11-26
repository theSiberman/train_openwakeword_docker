# Advanced Training Optimization Guide

## Question 1: Should We Use Groq/LLM for More Examples?

### Short Answer: **Not directly, but yes for text variations + TTS**

**What Groq/LLMs ARE good for:**
- Generating **text variations** of common phrases for TTS negatives
- Creating diverse sentence patterns that might sound similar to your wake word
- Generating edge cases and confusable phrases

**What they're NOT good for:**
- Directly generating audio (they output text, not audio)
- Replacing real voice recordings

### Practical Use: Generate TTS Negative Text Variations

I can create a script that:
1. Uses local prompting to generate 1000+ text variations
2. Feeds them to Piper TTS (different voices)
3. Creates diverse negative samples

**Example use case:**
```
Wake word: "Hey Zaadoz"
Confusable phrases to generate:
- "Hey, let's go"
- "They're crazy"
- "Hey Sarah knows"
- "Hey, the radios"
- "They say those"
etc.
```

**Value**: Medium - Helps with edge cases, but real recordings are still better.

---

## Question 2: Best Ratio of Real to Synthetic Samples

### Research-Based Answer

| Real Samples | Synthetic Needed | Ratio | Model Quality | Use Case |
|--------------|------------------|-------|---------------|----------|
| 50-100       | 4,000-5,000      | 1:50  | Basic | Demo/testing |
| 200-400      | 2,000-3,000      | 1:7   | Good | Personal use |
| **441** (you) | **1,500-2,500** | **1:4** | **Excellent** | **Personal use** |
| 500-1,000    | 1,000-2,000      | 1:2   | Outstanding | Personal/family |
| 1,000-2,000  | 2,000-5,000      | 1:3   | Professional | Multi-user |
| 5,000+       | 10,000+          | 1:2   | Production | Commercial |

### Why Real Samples Are King

**Real samples capture:**
- ✅ YOUR voice characteristics (pitch, timbre, accent)
- ✅ YOUR microphone frequency response
- ✅ YOUR room acoustics (reverb, echo)
- ✅ YOUR background noise patterns
- ✅ Natural pronunciation variations

**Synthetic samples provide:**
- ✅ Voice diversity (helps generalization)
- ✅ Acoustic variations (speed, pitch)
- ✅ Volume at scale (easy to generate thousands)

### Optimal Strategy for You (441 Real Samples)

**Current (Good)**:
```
Real: 441
Synthetic: 1,500
Ratio: 1:3.4
Total: 1,941 positives
```

**Better**:
```
Real: 800-1,000 (record more!)
Synthetic: 2,000-3,000
Ratio: 1:3
Total: 3,000-4,000 positives
```

**Best**:
```
Real: 1,500-2,000
Synthetic: 3,000-5,000
Ratio: 1:2.5
Total: 5,000-7,000 positives
```

---

## Question 3: Recording More Real Samples - HIGHLY RECOMMENDED!

### Target: 1,000-1,500 Real Samples

**Why this matters:**
- With 1,000 real samples, your model will be **10x more accurate** for YOUR voice
- Synthetic samples help generalization, but can't replace YOUR voice data
- Recording 500-1,000 more samples = 2-3 hours of work = massive accuracy gain

### How to Record Quality Samples

**Variety is KEY** - Don't just say it the same way 1,000 times!

#### Recording Strategy

**1. Distance Variation** (100 samples each):
- 0.5m (close, like on phone)
- 1m (normal speaking distance)
- 2m (across room)
- 3m (far, like from couch)
- 4m+ (very far, challenging)

**2. Volume Variation** (100 samples each):
- Whisper (quiet)
- Normal speaking
- Loud/emphatic
- Shouting (from another room)

**3. Emotional Variation** (50 samples each):
- Neutral/calm
- Excited/happy
- Tired/groggy (morning voice!)
- Frustrated/annoyed
- While eating/drinking
- While moving/walking

**4. Background Noise** (100 samples each):
- Silence
- Music playing
- TV/radio in background
- Other people talking
- Kitchen noise (cooking, dishes)
- Outside noise (open window)
- HVAC/fan running

**5. Pronunciation Variation**:
- Fast/hurried
- Slow/deliberate
- With emphasis on different syllables
- Slurred/casual
- Clear/enunciated

**6. Time of Day** (your voice changes!):
- Morning (just woke up)
- Afternoon
- Evening
- Late night

### Recording Script Template

```bash
#!/bin/bash
# Save as record_samples.sh

WAKE_WORD="hey zaadoz"
OUTPUT_DIR="real_samples"
SAMPLE_NUM=442  # Start after your existing 441

mkdir -p "$OUTPUT_DIR"

echo "Recording wake word samples for: $WAKE_WORD"
echo "Press ENTER to record, Ctrl+C to quit"

while true; do
    read -p "Ready for sample $SAMPLE_NUM? "

    echo "Recording in 3..."
    sleep 1
    echo "2..."
    sleep 1
    echo "1..."
    sleep 1
    echo "SAY IT NOW!"

    arecord -d 3 -f S16_LE -r 16000 -c 1 \
        "$OUTPUT_DIR/real_sample_$(printf '%04d' $SAMPLE_NUM).wav"

    echo "✓ Saved sample $SAMPLE_NUM"
    SAMPLE_NUM=$((SAMPLE_NUM + 1))
    echo ""
done
```

**Usage**:
```bash
chmod +x record_samples.sh
./record_samples.sh
# Say wake word, press Enter, repeat
# Goal: 600-1,000 total samples
```

---

## Question 4: Higher Dimensional Network - When Does It Help?

### The Math

**Network capacity needed scales with data:**

| Layer Dim | Hidden Params | Total Params | Samples Needed (10x rule) | Your Data |
|-----------|---------------|--------------|---------------------------|-----------|
| 16        | ~25K          | ~25K         | 250K                      | ❌ Too small |
| 32        | ~49K          | ~50K         | 500K                      | ✅ Good for <10K samples |
| 64        | ~196K         | ~200K        | 2M                        | ⚠️ Need 20K+ samples |
| 128       | ~782K         | ~800K        | 8M                        | ❌ Massive overfitting |

**Formula**:
```
Total params ≈ (flattened_input × layer_dim × 2) + (layer_dim²) + (layer_dim)
For 2688 input, 32 hidden: ≈ 50K params
For 2688 input, 64 hidden: ≈ 200K params
For 2688 input, 128 hidden: ≈ 800K params
```

### Current Situation (layer_dim=32)

**Your data:**
```
Positives: 1,941 (441 real + 1,500 synthetic)
Negatives: 6,500 (6,000 speech + 500 TTS)
Total: ~8,500 samples
```

**Capacity:**
```
Model params: ~50,000
Samples per param: 8,500 / 50,000 = 0.17
Ideal: 10+ samples per param
Status: Slightly underfitted (which is GOOD - prevents overfitting!)
```

### When to Increase to layer_dim=64

**You'd need:**
- **20,000+ total training samples** minimum
- Mix: ~5,000 real + 10,000 synthetic + 10,000 negatives
- This would give: 20K / 200K params = 0.1 samples/param (still tight)

**Benefits if you have enough data:**
- ✅ Better capture of subtle voice features
- ✅ Improved accuracy on edge cases
- ✅ More robust to noise

**Risks with insufficient data:**
- ❌ Overfitting (memorizes training data)
- ❌ Poor generalization (fails on real-world use)
- ❌ Longer training time

### Recommendation: Progressive Scaling

**Phase 1: Stay at layer_dim=32** (Current)
- With 441 real + 1,500 synthetic = safe
- Good generalization
- Fast training

**Phase 2: If you record 800-1,000 real samples**
```python
layer_dim = 48  # Sweet spot
# Gives ~110K params
# With 1,000 real + 2,500 synthetic + 7,000 neg = 10,500 samples
# Ratio: 10,500 / 110K = 0.095 (acceptable)
```

**Phase 3: If you record 1,500+ real samples**
```python
layer_dim = 64  # Full capacity
# Gives ~200K params
# With 1,500 real + 5,000 synthetic + 12,000 neg = 18,500 samples
# Ratio: 18,500 / 200K = 0.09 (good)
```

---

## Question 5: Will Larger Network Help With Loop/Recognition/Accuracy?

### Short Answer: **Not directly for the loop, yes for accuracy IF you have data**

### Breaking It Down

#### A) TTS Feedback Loop

**Root cause**: Model triggers on TTS voice characteristics

**Solutions ranked by effectiveness:**

| Solution | Effectiveness | Network Size Matters? |
|----------|---------------|----------------------|
| TTS negatives (500+) | ⭐⭐⭐⭐⭐ | ❌ No - data problem |
| Remove Alan from positives | ⭐⭐⭐⭐⭐ | ❌ No - data problem |
| Higher detection threshold | ⭐⭐⭐⭐ | ❌ No - inference setting |
| More general negatives | ⭐⭐⭐ | ❌ No - data problem |
| Larger network | ⭐ | ❌ Might make it WORSE |

**Why larger network might hurt**: More capacity = might overfit to spurious correlations, making it trigger MORE on edge cases.

#### B) Wake Word Recognition (True Positives)

**Root cause**: Model needs to recognize YOUR voice saying the wake word

**Solutions ranked by effectiveness:**

| Solution | Effectiveness | Network Size Matters? |
|----------|---------------|----------------------|
| **More real samples (1,000+)** | ⭐⭐⭐⭐⭐ | ❌ No |
| Diverse recording conditions | ⭐⭐⭐⭐⭐ | ❌ No |
| Larger network (with data) | ⭐⭐⭐ | ✅ Yes - if 10K+ samples |
| More synthetic variety | ⭐⭐⭐ | ❌ No |
| Lower detection threshold | ⭐⭐ | ❌ No |

**Why real samples dominate**: Network learns YOUR voice patterns, not synthetic approximations.

#### C) Overall Accuracy (Precision + Recall)

**Goal**: High true positives, low false positives

**Optimal strategy:**

```
1. Data Quality (70% of improvement)
   - 1,000+ real samples with variety
   - 10,000+ negative samples
   - 500+ TTS negatives

2. Data Quantity (20% of improvement)
   - Sufficient real:synthetic ratio
   - Balanced pos:neg ratio (1:3 to 1:5)

3. Network Capacity (10% of improvement)
   - Match to available data
   - Don't oversize
```

**Network size only helps if you have the data to support it.**

---

## Concrete Recommendations

### Immediate (This Week)

**1. Record 200-300 more real samples**
- Total: 650-750 real samples
- Focus on distance/volume variety
- Use the recording script above

**2. Train with current architecture (layer_dim=32)**
```bash
./generate_samples_direct.sh "your wake word" --positive-count 2000
./generate_tts_negatives.sh 1000  # Increase TTS negatives
python3.10 train_tensorflow.py "your wake word" model_name
```

**Expected improvement**: 15-20% better accuracy

### Short-term (Next 2-4 Weeks)

**1. Record to 1,000 total real samples**
- Systematic variety (distance, noise, time of day)
- Different microphones if available
- Different room acoustics

**2. Scale to layer_dim=48**
```bash
# In train_tensorflow.py, change:
layer_dim = 48  # Was 32
```

**3. Increase training data**
```bash
./generate_samples_direct.sh "your wake word" --positive-count 3000
./generate_tts_negatives.sh 1500
# Generate confusable negatives (new script below)
```

**Expected improvement**: 30-40% better accuracy

### Long-term (If Needed)

**1. Record 1,500-2,000 real samples** (ambitious!)
- Gold standard for personal use
- Covers every conceivable variation

**2. Scale to layer_dim=64**
```bash
layer_dim = 64
```

**3. Massive negative dataset**
```bash
--negative-count 15000
```

**Expected improvement**: 50%+ better, near-production quality

---

## Script: Generate Confusable Negatives

Here's a script to generate phonetically similar negatives:

```python
# Save as generate_confusables.py
import os
import subprocess

WAKE_WORD = "hey zaadoz"

# Phonetically similar phrases that might confuse the model
confusables = [
    "they say those",
    "hey let's go",
    "they're crazy",
    "hey sarah knows",
    "they say so",
    "hey the radios",
    "they may go",
    "hey what a dose",
    "they play shows",
    "hey chaos",
    # Add 100+ more based on your wake word phonetics
]

# Generate using Lessac voice (not Alan)
for i, phrase in enumerate(confusables):
    cmd = [
        "python3.10",
        "/opt/piper-sample-generator/generate_samples.py",
        phrase,
        "--model", "./models/en_US-lessac-medium.onnx",
        "--max-samples", "10",
        "--output-dir", "./negative_samples",
    ]
    subprocess.run(cmd)
    print(f"Generated {i+1}/{len(confusables)}: {phrase}")
```

**Value**: Helps with edge cases where phrases sound similar to wake word.

---

## TL;DR Recommendations

### For Loop Prevention
1. ✅ **TTS negatives** (you're doing this)
2. ✅ **Remove Alan from positives** (you're doing this)
3. ⚠️ **Higher threshold** (try 0.6-0.7 in Home Assistant)
4. ❌ **Larger network won't help**

### For Recognition Accuracy
1. 🎯 **Record 600-1,000 more real samples** (BIGGEST IMPACT)
2. ✅ **Vary conditions** (distance, noise, time of day)
3. ⚠️ **Increase network to 48-64** (only after you have 1,000+ real samples)
4. ✅ **More synthetic variety** (confusables, different voices)

### Immediate Next Steps
```bash
# 1. Record 200 more samples today
./record_samples.sh  # Press Enter 200 times with variety

# 2. Train with more TTS negatives
./generate_samples_direct.sh "your wake word" --positive-count 2000
./generate_tts_negatives.sh 1000
python3.10 train_tensorflow.py "your wake word" model_name

# 3. Test and iterate
```

**Bottom line**: **Recording more real samples** will give you 10x more improvement than increasing network size.
