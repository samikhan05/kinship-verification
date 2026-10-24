#!/usr/bin/env bash
# Install the extra packages the project needs on a Kaggle GPU notebook.
# Kept in the repo (not in the notebook) so that fixes arrive with `git pull`.
#
# * Kaggle's own torch / CUDA are kept: nothing here reinstalls torch.
# * onnxruntime-gpu 1.24.1: built for CUDA 12 (Kaggle), wheels for Python 3.11-3.14.
#   (Recent releases such as 1.31 target CUDA 13.)
# * onnx 1.21.0: wheels for Python 3.10-3.14, needs only protobuf >= 4.25.
#   (onnx 1.17 has no wheel for Python 3.13+ and fails to build from source.)
# * insightface and facenet-pytorch use --no-deps: insightface would add the
#   CPU-only `onnxruntime` (same module name as onnxruntime-gpu), and
#   facenet-pytorch 2.6.0 would downgrade torch to < 2.3. Their real dependencies
#   (numpy, opencv, scipy, scikit-image, requests, tqdm) are already on Kaggle.
set -e
python --version
pip uninstall -y -q onnxruntime 2>/dev/null || true
pip install -q onnxruntime-gpu==1.24.1 onnx==1.21.0
pip install -q insightface==2.1 --no-deps
pip install -q facenet-pytorch==2.6.0 --no-deps
echo "Install finished."
