#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

# 设置清华源
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.5.2" --no-cache-dir -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST && pip install torch-memory-saver --no-cache-dir -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST
fi
pip install --no-cache-dir "vllm==0.24.0" -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST

echo "2. install basic packages"
pip install "transformers==5.3.0" accelerate datasets peft hf-transfer \
    "numpy>=2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard \
    -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST

echo "3. install FlashAttention"
# Built from source rather than a prebuilt wheel, which is pinned to one torch
# release. FlashInfer is not installed here: vLLM pins the version it needs.
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
pip install --no-build-isolation flash_attn==2.8.3 -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1" -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi

echo "5. May need to fix opencv"
pip install opencv-python -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST
pip install opencv-fixer -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.10.2.21 -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST
fi

echo "Successfully installed all packages"