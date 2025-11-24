#!/bin/bash
# Direct setup without Docker - for RunPod or environments where Docker isn't available
set -e

echo "========================================"
echo "OpenWakeWord Training Setup (Direct)"
echo "========================================"
echo ""

VOICE_DIR="./models"

# 1. Download Piper voice models
echo "[1/3] Downloading Piper voice models..."
mkdir -p "$VOICE_DIR"

if [ -f "$VOICE_DIR/en_US-lessac-medium.onnx" ] && [ -f "$VOICE_DIR/en_US-lessac-medium.json" ]; then
    echo "  ✓ Voice models already exist, skipping download"
else
    echo "  → Downloading en_US-lessac-medium.onnx..."
    wget -q --show-progress -O "$VOICE_DIR/en_US-lessac-medium.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

    echo "  → Downloading en_US-lessac-medium.json..."
    wget -q --show-progress -O "$VOICE_DIR/en_US-lessac-medium.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

    # Also create the .onnx.json filename that piper expects
    cp "$VOICE_DIR/en_US-lessac-medium.json" "$VOICE_DIR/en_US-lessac-medium.onnx.json"

    echo "  ✓ Voice models downloaded successfully"
fi

# 2. Install Python packages
echo ""
echo "[2/3] Installing Python packages..."
echo "  This may take 5-10 minutes..."
python3.10 -m pip install -q openwakeword piper-tts webrtcvad onnx tensorflow torch torchinfo==1.8.0 torchmetrics==1.2.0 mutagen==1.47.0 scipy matplotlib datasets speechbrain

echo "  → Installing TFLite conversion tools..."
python3.10 -m pip install -q onnx2tf tf-keras onnx_graphsurgeon psutil flatbuffers

# 3. Clone and install piper-sample-generator
echo ""
echo "[3/3] Setting up piper-sample-generator..."
if [ ! -d "/opt/piper-sample-generator" ]; then
    git clone -q https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator
    cd /opt/piper-sample-generator && python3.10 -m pip install -q -e . && cd - > /dev/null
    echo "  ✓ Installed piper-sample-generator"
else
    echo "  ✓ piper-sample-generator already installed"
fi

# 4. Download OpenWakeWord models
echo ""
echo "Downloading OpenWakeWord models..."
python3.10 -c 'from openwakeword.utils import download_models; download_models()' > /dev/null 2>&1
echo "  ✓ OpenWakeWord models downloaded"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "You can now run ./train_direct.sh to train wake word models"
echo ""
