#!/bin/bash
# Direct setup without Docker - downloads models and installs dependencies
# For use on RunPod or any system with Python already installed

set -e

echo "========================================"
echo "OpenWakeWord Setup (Direct Install)"
echo "========================================"
echo ""

# --- Configuration ---
VOICE_DIR="./models"
NEGATIVE_DIR="./negative_samples"

# 1. Download Piper voice models (multiple voices for variety)
echo "[1/3] Downloading Piper voice models..."
mkdir -p "$VOICE_DIR"

# US English Female (Lessac - clear, medium quality)
if [ ! -f "$VOICE_DIR/en_US-lessac-medium.onnx" ]; then
    echo "  → Downloading en_US-lessac-medium (US Female)..."
    wget -q --show-progress -O "$VOICE_DIR/en_US-lessac-medium.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -q --show-progress -O "$VOICE_DIR/en_US-lessac-medium.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    cp "$VOICE_DIR/en_US-lessac-medium.json" "$VOICE_DIR/en_US-lessac-medium.onnx.json"
fi

# GB English Male (Alan - British)
if [ ! -f "$VOICE_DIR/en_GB-alan-medium.onnx" ]; then
    echo "  → Downloading en_GB-alan-medium (British Male)..."
    wget -q --show-progress -O "$VOICE_DIR/en_GB-alan-medium.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx
    wget -q --show-progress -O "$VOICE_DIR/en_GB-alan-medium.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
    cp "$VOICE_DIR/en_GB-alan-medium.json" "$VOICE_DIR/en_GB-alan-medium.onnx.json"
fi

# GB English Female (Alba - British)
if [ ! -f "$VOICE_DIR/en_GB-alba-medium.onnx" ]; then
    echo "  → Downloading en_GB-alba-medium (British Female)..."
    wget -q --show-progress -O "$VOICE_DIR/en_GB-alba-medium.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx
    wget -q --show-progress -O "$VOICE_DIR/en_GB-alba-medium.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json
    cp "$VOICE_DIR/en_GB-alba-medium.json" "$VOICE_DIR/en_GB-alba-medium.onnx.json"
fi

# US English Male (Libritts)
if [ ! -f "$VOICE_DIR/en_US-libritts-high.onnx" ]; then
    echo "  → Downloading en_US-libritts-high (US Male)..."
    wget -q --show-progress -O "$VOICE_DIR/en_US-libritts-high.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx
    wget -q --show-progress -O "$VOICE_DIR/en_US-libritts-high.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json
    cp "$VOICE_DIR/en_US-libritts-high.json" "$VOICE_DIR/en_US-libritts-high.onnx.json"
fi

echo "  ✓ Voice models ready"

# 2. Install Python packages
echo ""
echo "[2/3] Installing Python packages..."
echo "  This may take 5-10 minutes..."

# Determine Python command
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

$PYTHON_CMD -m pip install -q openwakeword piper-tts webrtcvad onnx onnx-tf onnxscript tensorflow torch torchinfo==1.8.0 torchmetrics==1.2.0 mutagen==1.47.0 scipy matplotlib datasets speechbrain soundfile librosa

echo "  ✓ Python packages installed"

# 3. Clone and install piper-sample-generator
echo ""
echo "[3/3] Setting up piper-sample-generator..."
if [ ! -d "/opt/piper-sample-generator" ]; then
    sudo mkdir -p /opt 2>/dev/null || mkdir -p /opt
    sudo git clone -q https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator 2>/dev/null || git clone -q https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator
    sudo chown -R $USER /opt/piper-sample-generator 2>/dev/null || true
    cd /opt/piper-sample-generator && $PYTHON_CMD -m pip install -q -e .
    cd - > /dev/null
    echo "  ✓ Installed piper-sample-generator"
else
    echo "  ✓ piper-sample-generator already installed"
fi

# Download OpenWakeWord models
echo "  → Downloading OpenWakeWord models..."
$PYTHON_CMD -c 'from openwakeword.utils import download_models; download_models()' >/dev/null 2>&1
echo "  ✓ OpenWakeWord models downloaded"

# Create directories
mkdir -p "$NEGATIVE_DIR"
mkdir -p generated_samples
mkdir -p real_samples

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Voice models downloaded:"
echo "  • en_US-lessac-medium (US Female)"
echo "  • en_GB-alan-medium (British Male)"
echo "  • en_GB-alba-medium (British Female)"
echo "  • en_US-libritts-high (US Male)"
echo ""
echo "Next steps:"
echo "  1. ./generate_samples_direct.sh \"your wake word\""
echo "  2. ./train_direct.sh \"your wake word\""
echo ""
echo "Optional: Add real recordings to ./real_samples/"
echo ""
