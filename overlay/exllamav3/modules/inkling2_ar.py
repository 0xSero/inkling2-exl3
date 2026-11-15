"""
Graph-safe PCIe P2P all-reduce (recursive doubling) for the small decode-step messages (see inkling2_ar.cu). Built on first use with
torch's JIT extension loader; if the build or the IPC setup fails, callers fall back to NCCL.

    ar = PeerAllReduce(rank, world, cap_bytes)    # collective: every rank constructs it at the same point
    ar(x)                                         # in place, fp16/bf16/fp32 contiguous, bytes % 16 == 0, bytes <= cap
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        from .inkling2_ar_src import SRC
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        bdir = os.environ.get("I2_AR_BUILD", os.path.join(cache, "inkling2_ar"))
        os.makedirs(bdir, exist_ok = True)
        src = os.path.join(bdir, "inkling2_ar.cu")
        if not os.path.exists(src) or open(src).read() != SRC:      # same source -> ninja keeps the cached build
            with open(src, "w") as f: f.write(SRC)
        _ext = load(name = "inkling2_ar_ext", sources = [src], build_directory = bdir, verbose = False,
                    extra_cuda_cflags = ["-O3", "--use_fast_math"])
    return _ext


class PeerAllReduce:
    def __init__(self, rank: int, world: int, cap_bytes: int = 2 << 20, blocks: int = 32, threads: int = 256):
        ext = _load() if rank == 0 else None
        dist.barrier()                    # rank 0 compiles first; the others load the cached build
        ext = ext or _load()
        self.ext, self.cap = ext, cap_bytes
        self.blocks = int(os.environ.get("I2_AR_BLOCKS", blocks))
        self.threads = int(os.environ.get("I2_AR_THREADS", threads))
        self.f32mid = int(os.environ.get("I2_AR_F32MID", 0))  # 1: round-2 partials in fp32 (no intermediate rounding; T=48 37 vs 26 us)
        self.gb = int(os.environ.get("I2_AR_GB", 1))        # per-GPU grid barrier between rounds (one P2P destination at a time)
        self.h = ext.init(rank, world, cap_bytes)
        mine = ext.handles(self.h)
        allh = [None] * world
        dist.all_gather_object(allh, mine)
        ext.open(self.h, torch.stack(allh))
        dist.barrier()

    @classmethod
    def create(cls, rank: int, world: int, cap_bytes: int):
        """collective; returns None on every rank (callers use NCCL) if the build or the IPC setup fails on any rank"""
        ok = torch.ones(1, device = torch.device("cuda", torch.cuda.current_device()))
        try:
            if rank == 0: _load()
        except Exception as e:
            print(f"inkling2_ar: build failed ({e}); using NCCL", flush = True); ok.zero_()
        dist.all_reduce(ok, op = dist.ReduceOp.MIN)
        if ok.item() == 0: return None
        try:
            return cls(rank, world, cap_bytes)
        except Exception as e:      # IPC/P2P setup failure: every rank must reach this point to agree
            print(f"inkling2_ar: setup failed on rank {rank} ({e}); using NCCL", flush = True)
            return None

    def fits(self, x: torch.Tensor) -> bool:
        n = x.numel() * x.element_size()
        return n * 2 <= self.cap and n % 16 == 0 and x.is_contiguous()

    def big(self, x: torch.Tensor) -> torch.Tensor:
        """reduce-scatter + all-gather path (W = 4) for messages up to 2 x cap bytes"""
        self.ext.all_reduce_big(self.h, x, self.threads, int(os.environ.get("I2_AR_GB", 1)))
        return x

    def fits_big(self, x: torch.Tensor) -> bool:
        n = x.numel() * x.element_size()
        return n // 2 <= self.cap and n % 64 == 0 and x.is_contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.ext.all_reduce(self.h, x, self.blocks, self.threads, self.f32mid, self.gb)
        return x
