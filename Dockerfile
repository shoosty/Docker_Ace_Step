FROM --platform=linux/amd64 runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# ACE-Step 1.5 XL — v59
# PyTorch 2.8, CUDA 12.8, official ACE-Step 1.5 requirements
# v56: silences the worker-startup warnings + enables the upgrades
#      Stephen flagged as "LoRA is the moat, speed is the margin":
#        + lightning + bitsandbytes + lycoris-lora  (LoKr LoRA stack)
#        + nano-vllm                                 (LM speedup)
#        + pytorch_wavelets + PyWavelets             (DCW quality filter)
#      flash_attn intentionally still NOT installed per Stephen's
#      "until this is solid" rule.
# v57: handler-only change — size-aware WAV upload. 12-min songs
#      blow past Supabase's 50 MB per-object cap; v57 skips the WAV
#      and returns the song from the MP3 alone instead of failing
#      the whole job. No deps changed, but bumping the tag so the
#      endpoint cache flips.
# v58: handler-only — pass full quality params + activate DCW. The
#      DCW activation turned out to be the quality regression that
#      sent us into a 9-test recipe search; see handler.py docstring.
# v59: handler-only change — bake the sweet-spot recipe defaults so
#      the studio stops needing to send overrides on every request.
#      dcw_enabled=False, guidance_scale=10.0 (everything else
#      unchanged). No image-side surgery.
#      Known unsolved (deferred to v60 with image rebuild):
#        + flash_attn import fails on RTX 5090 + CUDA 12.8 (symbol
#          mismatch — needs a CUDA-12.8-compatible wheel rebuild).
#          Currently falls back to PyTorch sdpa; works but slower.
#        + nano-vllm fails on the same hardware due to triton's
#          `triton_key` having been removed in a later version.
#          Pin a compatible triton wheel to restore the LM speedup.

RUN apt-get update && apt-get install -y git curl ffmpeg && apt-get clean

RUN mkdir /ace-step-code && \
    git clone https://github.com/ace-step/ACE-Step-1.5.git /ace-step-code

COPY patch_config.py /patch_config.py
RUN python3 /patch_config.py

RUN pip install -e /ace-step-code --no-deps && \
    pip install fsspec jinja2 networkx sympy setuptools \
        diffusers "transformers>=4.51.0,<4.58.0" accelerate peft soundfile \
        librosa loguru tqdm numpy click "datasets==3.4.1" \
        "pytorch_lightning==2.5.1" lightning \
        pypinyin num2words py3langid \
        hangul-romanize spacy thinc hf_transfer \
        einops "vector_quantize_pytorch>=1.27.15" \
        bitsandbytes \
        lycoris-lora \
        pytorch_wavelets PyWavelets \
        runpod torchcodec typing_extensions \
        supabase && \
    pip install "click>=8.0"

# v56 — nano-vllm ships as a submodule inside the ACE-Step repo.
# Install it as its own layer so future pip-line edits don't
# invalidate this build step.
RUN pip install /ace-step-code/acestep/third_parts/nano-vllm

RUN mkdir -p /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
