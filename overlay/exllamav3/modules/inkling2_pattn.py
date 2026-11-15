"""
SM120 prefill attention for Inkling-2 (one TP rank: 12 query heads over 2 KV heads, head dim 128), CUDA mma.sync kernels in
inkling2_pattn_src.py. Global layers: paged 4-bit q16 rotated KV (ng = 4); sliding-window layers: int8 ring. Same math as
inkling2_ops.attention (relative bias, sinks, gate, rotation back), fp32 softmax / accumulation, fp16 K/V/P operands.

    pa = PrefillAttn(device)
    pa.run(kind, q, rel_q, rel_proj_bf16, sink, gate, kq, vq, ks, vs, page_table, ring, seqs, out)
      kind 0 = global, 1 = SWA; seqs = [(first batch row, rows, position of first row, slot), ...] (host ints)
"""
from __future__ import annotations
import os, math
import torch

_ext = None
BM, BN, GQ = 128, 64, 6


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        from .inkling2_pattn_src import CUDA_SRC, CPP_SRC
        bdir = os.environ.get("I2_PATTN_BUILD", os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "inkling2_pattn"))
        os.makedirs(bdir, exist_ok = True)
        _ext = load_inline(name = "inkling2_pattn_ext", cpp_sources = [CPP_SRC], cuda_sources = [CUDA_SRC], functions = ["pattn"],
                           build_directory = bdir, verbose = os.environ.get("I2_PATTN_VERBOSE") == "1",
                           extra_cuda_cflags = ["-O3", "-gencode=arch=compute_120,code=sm_120", "-lineinfo",
                                                "--ptxas-options=-v" if os.environ.get("I2_PATTN_VERBOSE") == "1" else "-DNOP"])
    return _ext


def permuted_rotation(rot_bf16):
    """fp16 [128, 128] with rot[j, slot] = M[j, perm(slot)]: the kernel keeps head dims in slot order (2t -> t, 2t+1 -> t+4 per 8)"""
    s = torch.arange(128)
    r = s % 8
    perm = (s // 8) * 8 + torch.where(r % 2 == 0, r // 2, r // 2 + 4)
    return rot_bf16.float()[:, perm.to(rot_bf16.device)].half().contiguous()


class PrefillAttn:
    def __init__(self, device, min_seg = 4):
        self.ext = _load()
        self.dev = torch.device(device)
        self.nsm = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self.min_seg = min_seg
        self.part_o = torch.empty((self.nsm, 2, BM, 128), dtype = torch.float32, device = self.dev)
        self.part_ml = torch.empty((self.nsm, 2, BM, 2), dtype = torch.float32, device = self.dev)
        self._rot = {}
        self._relpT = {}
        self._dummy = torch.zeros(16, dtype = torch.int32, device = self.dev)
        self._plan_cache = {}

    @staticmethod
    def _cached(cache, src, make):
        # keyed by address, but the source tensor is kept alive alongside, so a freed tensor's address cannot alias
        e = cache.get(src.data_ptr())
        if e is None or e[0] is not src:
            if len(cache) > 256:
                cache.clear()
            e = cache[src.data_ptr()] = (src, make(src))
        return e[1]

    def rot(self, rot_bf16):
        return self._cached(self._rot, rot_bf16, permuted_rotation)

    def relpT(self, rel_proj):
        return self._cached(self._relpT, rel_proj, lambda t: t.to(torch.bfloat16).t().contiguous())

    def plan(self, kind, seqs, hkl):
        key = (kind, tuple(seqs), hkl)
        p = self._plan_cache.get(key)
        if p is not None:
            return p
        units, cum = [], [0]
        for (t0, n, p0, slot) in seqs:
            F = n * GQ
            for f0 in range(0, F, BM):
                nf = min(BM, F - f0)
                p_lo, p_hi = p0 + f0 // GQ, p0 + (f0 + nf - 1) // GQ
                b_hi = p_hi // BN
                b_lo = 0 if kind == 0 else max(0, p_lo - 1023) // BN
                for hk in range(hkl):
                    units.append((t0, p0, f0, nf, hk, b_lo, b_hi, slot))
                    cum.append(cum[-1] + b_hi - b_lo + 1)
        total = cum[-1]
        ncta = max(1, min(self.nsm, -(-total // self.min_seg)))
        seg = -(-total // ncta)
        ncta = -(-total // seg)
        ut = torch.tensor(units, dtype = torch.int32).pin_memory().to(self.dev, non_blocking = True)
        ct = torch.tensor(cum, dtype = torch.int32).pin_memory().to(self.dev, non_blocking = True)
        p = (ut, ct, len(units), seg, total, ncta)
        if len(self._plan_cache) > 256:
            self._plan_cache.clear()
        self._plan_cache[key] = p
        return p

    def add_lane(self):
        self.part_o1 = torch.empty_like(self.part_o)
        self.part_ml1 = torch.empty_like(self.part_ml)

    def run(self, kind, q, rel_q, rel_proj, sink, gate, kq, vq, ks, vs, page_table, ring, seqs, out, rot = None, lane = 0):
        hkl = kq.shape[-2]
        ut, ct, nu, seg, total, ncta = self.plan(kind, seqs, hkl)
        if kind == 1:
            assert ring % BN == 0
        self.ext.pattn(kind, q, rel_q, self.relpT(rel_proj), sink, gate, kq, vq, ks, vs,
                       page_table if page_table is not None else self._dummy, ring,
                       self.rot(rot) if rot is not None else self._dummy, ut, ct, nu, seg, total, ncta, out, *((self.part_o1, self.part_ml1) if lane else (self.part_o, self.part_ml)))
        return out
