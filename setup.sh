#!/bin/bash
# Setup script for OpenWakeWord training environment
# Run this once to prepare the Docker image with all dependencies

set -e

echo "========================================"
echo "OpenWakeWord Training Environment Setup"
echo "========================================"
echo ""

# --- Configuration ---
VOICE_DIR="./models"
CUSTOM_IMAGE="oww-training-env"

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

    echo "  ✓ Voice models downloaded successfully"
fi

# 2. Build custom Docker image with dependencies
echo ""
echo "[2/3] Building custom Docker image with dependencies..."
echo "  This may take 5-10 minutes..."
echo ""

docker run --name oww-setup-temp \
    --entrypoint /bin/bash \
    rhasspy/wyoming-openwakeword \
    -c "
        set -e
        echo '  → Installing system packages...'
        apt-get update -qq && apt-get install -y -qq git wget build-essential python3-dev > /dev/null

        echo '  → Installing Python packages...'
        pip install --break-system-packages -q openwakeword piper-tts piper-phonemize webrtcvad onnx onnx-tf onnxscript torchinfo==1.8.0 torchmetrics==1.2.0 mutagen==1.47.0 scipy matplotlib datasets speechbrain

        echo '  → Cloning piper-sample-generator...'
        git clone -q https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator

        echo '  → Installing piper-sample-generator...'
        cd /opt/piper-sample-generator && pip install --break-system-packages -q -e .

        echo '  → Downloading OpenWakeWord models...'
        python3 -c 'from openwakeword.utils import download_models; download_models()' >/dev/null 2>&1

        echo '  ✓ All dependencies installed'
    "

echo ""
echo "  → Committing container to image '$CUSTOM_IMAGE'..."
docker commit oww-setup-temp $CUSTOM_IMAGE > /dev/null

echo "  → Cleaning up temporary container..."
docker rm oww-setup-temp > /dev/null

echo "  ✓ Custom Docker image created successfully"

# 3. Verify installation
echo ""
echo "[3/3] Verifying installation..."

docker run --rm $CUSTOM_IMAGE /bin/bash -c "
    python3 -c 'import openwakeword; import piper; print(\"  ✓ Python packages OK\")' && \
    test -f /opt/piper-sample-generator/generate_samples.py && echo '  ✓ piper-sample-generator OK'
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "You can now run ./train.sh to train wake word models"
echo "The setup image '$CUSTOM_IMAGE' is ready for fast iteration"
echo ""
