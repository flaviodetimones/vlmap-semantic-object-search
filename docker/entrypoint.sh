#!/bin/bash
set -e

# Set XDG_RUNTIME_DIR for Open3D / GUI applications
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"

# Activate the tfg conda environment for interactive shells
source /opt/conda/etc/profile.d/conda.sh
conda activate tfg

# ── Menu: symlink to the volume-mounted script (always up to date) ────────────
chmod +x /workspace/docker/menu.sh
ln -sf /workspace/docker/menu.sh /usr/local/bin/menu

# ── Ensure vlmaps is always importable ───────────────────────────────────────
grep -qxF 'export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH' /root/.bashrc \
    || echo 'export PYTHONPATH=/workspace/third_party/vlmaps:$PYTHONPATH' >> /root/.bashrc

# ── Alias OPENAI_KEY for scripts that expect it instead of OPENAI_API_KEY ────
grep -qxF 'export OPENAI_KEY=$OPENAI_API_KEY' /root/.bashrc \
    || echo 'export OPENAI_KEY=$OPENAI_API_KEY' >> /root/.bashrc
export OPENAI_KEY=$OPENAI_API_KEY

# ── Welcome banner ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         TFG — VLMaps Semantic Object Search                     ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Python  : $(python --version 2>&1)"
echo "║  CUDA    : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "║  GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'No GPU detected')"
echo "║  Workdir : $(pwd)"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Type  'menu'  to see available commands                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

exec "$@"
