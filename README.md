# OpenWakeWord Custom Training

Train custom wake word models for use with OpenWakeWord and Home Assistant.

## Features

- **Multiple Voice Models**: US & British English, male & female voices for variety
- **Acoustic Variations**: Automatic speed, pitch, and noise variations
- **Real Sample Support**: Mix your own recordings with synthetic samples
- **Negative Samples**: Trained with background speech, noise, and music (6,000 samples)
- **Optimal Training**: 3,000 positive + 6,000 negative samples (based on OpenWakeWord best practices)
- **TFLite Export**: Ready for Home Assistant deployment

## Quick Start

### 1. Setup (one-time)

```bash
./setup_direct.sh
```

Downloads:
- 4 Piper TTS voice models (US/British, male/female)
- OpenWakeWord feature extraction models
- Python dependencies (PyTorch, TensorFlow, etc.)
- piper-sample-generator tool

### 2. Generate Training Samples

```bash
./generate_samples_direct.sh "your wake word"
```

Optional parameters:
- `--positive-count 3000` - Number of positive samples (default: 3000)
- `--negative-count 6000` - Number of negative samples (default: 6000)

Example:
```bash
./generate_samples_direct.sh "hey jarvis" --positive-count 5000
```

This will:
- Generate 3,000 synthetic samples using 4 different voices
- Download 6,000 negative samples (speech, noise, music)
- Apply acoustic variations (speed, pitch, noise)
- Mix any existing samples from `./real_samples/`

### 3. Train the Model

```bash
./train_direct.sh "your wake word" [output_name]
```

Example:
```bash
./train_direct.sh "hey jarvis" hey_jarvis
```

Outputs:
- `hey_jarvis.onnx` - ONNX format model
- `hey_jarvis.tflite` - TFLite format for Home Assistant

## Adding Real Recordings

To improve accuracy, add your own recordings:

1. Record yourself saying the wake word (16kHz WAV format)
2. Save files to `./real_samples/`
3. Run `./generate_samples_direct.sh "your wake word"`
   - Script will use real samples + generate synthetic to reach 3,000 total
4. Train as normal: `./train_direct.sh "your wake word"`

## Workflow

```
┌─────────────────┐
│  setup_direct.sh │  (one-time setup)
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│ generate_samples_direct.sh│  (generate training data)
└────────┬─────────────────┘
         │
         ▼
┌─────────────────┐
│ train_direct.sh │  (train & export model)
└────────┬────────┘
         │
         ▼
    hey_jarvis.tflite  (deploy to Home Assistant)
```

## Sample Counts

Based on OpenWakeWord documentation and community best practices:

- **Positive samples**: 3,000 (mix of synthetic + real)
  - Distributed across 4 voices for variety
  - Acoustic variations applied automatically

- **Negative samples**: 6,000 (2:1 ratio)
  - 60% speech (Common Voice dataset)
  - 20% environmental noise (ESC-50 dataset)
  - 20% music (GTZAN dataset)

## Directory Structure

```
train-oww/
├── setup_direct.sh              # One-time setup
├── generate_samples_direct.sh   # Generate training data
├── train_direct.sh              # Train model
├── models/                      # Voice models (gitignored)
├── generated_samples/           # Synthetic samples (gitignored)
├── real_samples/                # Your recordings (gitignored)
├── negative_samples/            # Background audio (gitignored)
└── *.tflite                     # Output models (gitignored)
```

## Using with Home Assistant

1. Copy the `.tflite` file to Home Assistant:
   ```bash
   scp hey_jarvis.tflite homeassistant:/config/custom_components/openwakeword/models/
   ```

2. Configure in Home Assistant:
   ```yaml
   openwakeword:
     models:
       - hey_jarvis
   ```

3. Restart Home Assistant

## Troubleshooting

### "No positive samples found"
- Run `./generate_samples_direct.sh "your wake word"` first

### "TFLite conversion failed"
- Check if you have TensorFlow installed: `pip show tensorflow`
- On older CPUs without AVX, use RunPod or a newer machine

### Improving Accuracy
1. Add 100+ real recordings to `./real_samples/`
2. Increase sample counts: `--positive-count 5000 --negative-count 10000`
3. Record real false activations and add to `negative_samples/`

## Technical Details

- **Feature Extraction**: OpenWakeWord's embedding model (768-dim features)
- **Model Architecture**: 2-layer neural network (64 hidden units, dropout)
- **Training**: 100 epochs, Adam optimizer, binary cross-entropy loss
- **Acoustic Variations**:
  - Slerp weights: 0.3 (speaker blending)
  - Length scales: 0.8, 1.0, 1.2 (speed variation)
  - Noise scales: 0.4, 0.667, 0.8 (natural variation)

## Resources

- [OpenWakeWord GitHub](https://github.com/dscripka/openWakeWord)
- [OpenWakeWord Training Notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator)
