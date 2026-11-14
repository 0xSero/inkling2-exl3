"""In-image import check (no GPU memory used): the exllamav3 extension loads and the b12x kernels import.
Run: docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= ghcr.io/0xsero/inkling2-exl3:latest check"""
import os, sys, importlib.metadata as md

import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda})")

from exllamav3.ext import exllamav3_ext as ext
print(f"exllamav3 extension: {ext.__file__}")
import exllamav3.modules.inkling2_engine, exllamav3.modules.inkling2_draft  # noqa: F401
from exllamav3.modules import inkling2_b12x
print(f"Inkling-2 overlay modules import; b12x adapter tree {inkling2_b12x._FI}")

fns = inkling2_b12x._import_b12x()
print("b12x kernels import:", ", ".join(f.__name__ for f in fns))
import b12x
print(f"b12x package: {os.path.dirname(b12x.__file__)}")
commit = open(os.path.join(inkling2_b12x._FI, "COMMIT")).read().strip() if os.path.exists(os.path.join(inkling2_b12x._FI, "COMMIT")) else "?"
print(f"FlashInfer tree commit {commit}")
for p in ("nvidia-cutlass-dsl", "cuda-python", "apache-tvm-ffi", "triton"):
    print(f"{p} {md.version(p)}")
print("I2_B12X", os.environ.get("I2_B12X"), "EXL3_LOAD_ARENA", os.environ.get("EXL3_LOAD_ARENA"))
print("OK")
