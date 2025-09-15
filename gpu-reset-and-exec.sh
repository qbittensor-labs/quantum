#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"

# turn MIG off, try device reset (non-sudo then sudo -n), and fallback toggles.
nvidia-smi -i "$GPU_ID" -mig 0 || true
nvidia-smi --gpu-reset -i "$GPU_ID" || true
sudo -n nvidia-smi --gpu-reset -i "$GPU_ID" || true
sudo -n rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia || true
sudo -n modprobe nvidia || true
nvidia-smi --gpu-reset -i "$GPU_ID" || true
sudo -n nvidia-smi --gpu-reset -i "$GPU_ID" || true
nvidia-smi -i "$GPU_ID" -mig 1 || true
nvidia-smi -i "$GPU_ID" -mig 0 || true

exec "$@"

