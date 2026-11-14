"""In-image import check (no GPU memory used): the exllamav3 extension loads and the b12x kernels import.
Run: docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= ghcr.io/0xsero/inkling2-exl3:latest check"""
import os, sys, importlib.metadata as md

import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda})")

from exllamav3.ext import exllamav3_ext as ext
print(f"exllamav3 extension: {ext.__file__}")
import exllamav3.modules.inkling2_engine, exllamav3.modules.inkling2_draft  # noqa: F401
import exllamav3.modules.inkling2_gemm, exllamav3.modules.inkling2_markov  # noqa: F401  skinny GEMM / FP8 draft, fused Markov head
from exllamav3.modules import inkling2_b12x
assert all(hasattr(inkling2_b12x, n) for n in ("prepare_model", "attach_moe", "B12xMoEShard", "B12xMoE"))
print(f"Inkling-2 overlay modules import; b12x adapter tree {inkling2_b12x._FI}; expert layout {inkling2_b12x.ep_layout()}")
for f in ("/opt/inkling2/server/i2_server.py", "/opt/inkling2/bench/i2_sweep.py"):
    compile(open(f, encoding="utf-8").read(), f, "exec")
print("server and sweep compile")

fns = inkling2_b12x._import_b12x()
print("b12x kernels import:", ", ".join(f.__name__ for f in fns))
import b12x
print(f"b12x package: {os.path.dirname(b12x.__file__)}")
commit = open(os.path.join(inkling2_b12x._FI, "COMMIT")).read().strip() if os.path.exists(os.path.join(inkling2_b12x._FI, "COMMIT")) else "?"
print(f"FlashInfer tree commit {commit}")
for p in ("nvidia-cutlass-dsl", "cuda-python", "apache-tvm-ffi", "triton"):
    print(f"{p} {md.version(p)}")
print(" ".join(f"{k}={os.environ.get(k)}" for k in ("I2_B12X", "I2_EP_LAYOUT", "I2_DRAFT_FP8", "I2_MOE_GROUP", "EXL3_LOAD_ARENA")))
print("OK")
