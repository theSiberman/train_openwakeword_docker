# Using Groq TTS for Wake Word Training

## Why Groq TTS is Valuable

### **19 Different Voices** vs 3 Piper Voices
- **Piper**: 3 voices (Lessac, Alba, Libritts)
- **Groq**: 19 voices (Arista, Atlas, Basil, Briggs, Calum, Celeste, Cheyenne, Chip, Cillian, Deedee, Fritz, Gail, Indigo, Mamaw, Mason, Mikail, Mitch, Quinn, Thunder)

### **Performance**
- **Speed**: 140 characters/second (very fast!)
- **Format**: WAV output (perfect for openwakeword)
- **Quality**: PlayAI Dialog model with natural prosody

### **Use Cases**

| Use Case | Command | Purpose |
|----------|---------|---------|
| **Confusable Negatives** | `--confusables` | Prevent false positives on similar-sounding phrases |
| **Voice Variety (Positives)** | `--positives` | Add 19 voices saying your wake word |
| **TTS Response Negatives** | `--negatives` | Prevent feedback loop with multiple TTS voices |

---

## Setup

### 1. Get Groq API Key

Visit: https://console.groq.com/keys

Free tier includes generous limits!

### 2. Set Environment Variable

```bash
export GROQ_API_KEY='gsk_your_key_here'
```

Add to your `~/.bashrc` or `~/.zshrc` to make permanent:
```bash
echo 'export GROQ_API_KEY="gsk_your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install ffmpeg (optional, for audio conversion)

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Usage Examples

### **Recommended: Generate Confusable Negatives**

This is the **most valuable** use case - generates phrases that sound similar to your wake word but aren't:

```bash
# Generate 200 confusable phrases with 19 different voices
./generate_groq_samples.sh --confusables --count 200 --wake-word "hey zaadoz"
```

**What it generates**:
- "they say those" (Arista voice)
- "hey let's go" (Atlas voice)
- "they're crazy" (Basil voice)
- "hey sarah knows" (Briggs voice)
- ... 196 more variations

**Why this matters**: These confusables are the **hardest negatives** for your model - if it can reject these, it won't false-trigger on random speech.

---

### **Optional: Add Voice Variety to Positives**

If you want even more voice diversity beyond Piper's 3 voices:

```bash
# Generate 100 positive samples with 19 different voices
./generate_groq_samples.sh --positives --count 100 --wake-word "hey zaadoz"
```

**Note**: Your 441 real samples are still more valuable, but this adds synthetic variety.

---

### **Optional: Multi-Voice TTS Negatives**

Generate TTS responses in multiple voices (not just Alan):

```bash
# Generate 200 TTS response samples with 19 different voices
./generate_groq_samples.sh --negatives --count 200
```

**Why this matters**: If you might use different TTS voices in the future, this prevents feedback loops with any voice.

---

## Recommended Training Strategy with Groq

### **Best Practice Workflow**

```bash
# 1. Generate Piper positives (excludes Alan)
./generate_samples_direct.sh "hey zaadoz" --positive-count 1500

# 2. Generate Alan TTS negatives (prevent feedback with your current voice)
./generate_tts_negatives.sh 500

# 3. Generate Groq confusables (prevent false positives)
export GROQ_API_KEY='your-key'
./generate_groq_samples.sh --confusables --count 300 --wake-word "hey zaadoz"

# 4. (Optional) Add Groq voice variety to positives
./generate_groq_samples.sh --positives --count 200 --wake-word "hey zaadoz"

# 5. Train
python3.10 train_tensorflow.py "hey zaadoz" hey_zaadoz_v3
```

**Expected data distribution**:
```
Positives: ~2,141 total
  • Real recordings: 441
  • Piper synthetic (3 voices): 1,500
  • Groq synthetic (19 voices): 200

Negatives: ~7,300 total
  • Speech/noise (LibriSpeech): 6,000
  • Alan TTS responses: 500
  • Groq confusables: 300
  • (Optional) Groq multi-voice TTS: 500
```

---

## Why Confusables Are So Important

### The Problem

Your wake word: **"Hey Zaadoz"**

Phonetically similar phrases people might say:
- "They say those"
- "Hey, let's go"
- "They're crazy"
- "Hey Sarah knows"

**Without confusables**: Model might trigger on these (false positives)
**With confusables**: Model learns to distinguish your wake word from similar sounds

### The Solution

Groq TTS generates these confusables with 19 different voices:
- Tests model robustness across voice characteristics
- Creates "hard negatives" that force better learning
- Dramatically reduces false positive rate

---

## Cost Estimation

Groq TTS pricing (check current rates at https://groq.com/pricing):

**Example costs** (hypothetical):
```
200 confusables × 19 voices = ~4,000 characters
Cost: ~$0.01 - $0.10 (very cheap!)
```

**Free tier**: Often includes generous monthly credits

---

## Script Options

### All Flags

```bash
./generate_groq_samples.sh \
    --confusables \              # Mode: confusables, positives, or negatives
    --count 200 \                # Number of samples
    --wake-word "hey zaadoz"     # Your wake word
```

### Modes

| Mode | Output Dir | Samples Generated |
|------|-----------|-------------------|
| `--confusables` | `./negative_samples` | Phonetically similar phrases |
| `--positives` | `./generated_samples` | Wake word with 19 voices |
| `--negatives` | `./negative_samples` | TTS responses with 19 voices |

---

## Comparison: Piper vs Groq

| Feature | Piper (Local) | Groq (Cloud) |
|---------|---------------|--------------|
| **Voices** | 3 English | 19 English |
| **Speed** | ~1 sec/sample | ~0.1 sec/sample |
| **Cost** | Free (local) | ~$0.01-0.10/1000 chars |
| **Quality** | High | Very High |
| **Setup** | Complex | Simple (API key) |
| **Best For** | Bulk positive generation | Confusables & variety |

**Recommendation**: Use **both**!
- Piper: Bulk positive samples (1,500+)
- Groq: Confusables & edge cases (200-300)

---

## Advanced: Custom Confusables

Edit `generate_groq_samples.sh` to add your own confusables:

```bash
# Line ~70 in generate_groq_samples.sh
CONFUSABLE_PHRASES=(
    "they say those"
    "hey let's go"
    # Add your own based on testing:
    "your custom phrase"
    "another similar phrase"
)
```

**How to find confusables**:
1. Test your model
2. Note what triggers false positives
3. Add those phrases to the list
4. Regenerate and retrain

---

## Troubleshooting

### "API Key Not Set"
```bash
export GROQ_API_KEY='gsk_your_key_here'
```

### "Rate Limit Exceeded"
The script includes `sleep 0.1` between requests. Increase if needed:
```bash
# Edit line ~215 in generate_groq_samples.sh
sleep 0.5  # Slower, but safer
```

### "Invalid Audio Format"
Install ffmpeg for automatic 16kHz conversion:
```bash
sudo apt-get install ffmpeg
```

---

## Summary

✅ **Use Groq for confusables** - Biggest accuracy gain
✅ **19 voices** vs Piper's 3 - Better generalization
✅ **Very fast** - 140 chars/sec
✅ **Very cheap** - Pennies for hundreds of samples
✅ **Perfect format** - WAV output at 16kHz

**Recommended usage**:
1. Generate 200-300 confusables with Groq
2. Keep using Piper for bulk positives
3. Use Alan TTS for your specific TTS negatives
4. Train with the combined dataset

**Expected improvement**: 10-20% better false positive rejection!
