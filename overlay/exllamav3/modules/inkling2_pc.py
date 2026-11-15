"""
Inkling-2 prefill collectives on the copy engines (prefill-comm). Ring all-gather and ring reduce-scatter over
IPC-mapped slabs (inkling2_pc_src.py): bytes move with cudaMemcpyAsync peer copies on a dedicated copy stream; flags are
cuStreamWriteValue32 / cuStreamWaitValue32 stream memory operations, so waiting and moving hold no SMs (the CE ring
idea follows b12x comm/pcie/pcie_dma.py, Local Inference Lab / FlashInfer, Apache-2.0). Only the reduce-scatter
adds run on SMs (one fused sum kernel per step).

Ring: rank r sends to r + 1 and receives from r - 1, one P2P flow per GPU and direction.
  all-gather   step k: forward block (r - k) mod W (own block at k = 0), received at step k - 1.
  reduce-scatter step k: send (partial + received) of block (r - 1 - k) mod W; the owner adds its own partial last.
Flags per collective id (cid): data[k] (receiver side, set by r - 1 after its step-k copy) and ack[k] (sender side, set
by r + 1 after it consumed step k), epochs counted on the host (every rank issues the same sequence of collectives).
Receive slots are double-buffered by epoch parity and a sender waits for the ack of epoch e - 2, so no rank can
overwrite a slot its peer still reads. Eager only (host epochs), prefill path.

    pc = PcComm.create(rank, world, slab_bytes)
    buf = pc.region("x", (T, D), torch.bfloat16)           # slab region at the same offset on every rank
    pc.all_gather("x", buf, blocks)                         # blocks[o] = (row0, rows) of owner o's rows in buf
    pc.reduce_scatter("attn", partial, blocks, out)         # out[:rows_r] = sum over ranks of partial[blocks[r]]
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist

_ext = None
_DT = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2, torch.int32: 3, torch.uint8: 4}


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        from .inkling2_pc_src import SRC
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        bdir = os.environ.get("I2_PC_BUILD", os.path.join(cache, "inkling2_pc"))
        os.makedirs(bdir, exist_ok = True)
        src = os.path.join(bdir, "inkling2_pc.cu")
        if not os.path.exists(src) or open(src).read() != SRC:
            with open(src, "w") as f: f.write(SRC)
        _ext = load(name = "inkling2_pc_ext", sources = [src], build_directory = bdir, verbose = False,
                    extra_cuda_cflags = ["-O3"], extra_ldflags = ["-lcuda"])
    return _ext


NSTEP = 8           # flag slots per collective id and kind (>= world - 1)


class PcComm:
    def __init__(self, rank, world, slab_bytes, max_cids = 64, rs_rows = 0):
        ext = _load() if rank == 0 else None
        dist.barrier()
        self.ext = ext or _load()
        self.rank, self.world = rank, world
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.max_cids = max_cids
        self.nflag = max_cids * 2 * NSTEP
        self.h = self.ext.init(rank, world, slab_bytes, self.nflag)
        allh = [None] * world
        dist.all_gather_object(allh, self.ext.handles(self.h))
        self.ext.open(self.h, torch.stack(allh))
        dist.barrier()
        self.cap, self.top = slab_bytes, 0
        self.rs_rows = rs_rows          # receive-slot rows of every reduce-scatter (0: the first call's block rows)
        self.regions = {}               # name -> (offset, tensor)
        self.cids, self.epoch = {}, {}
        self.cs = torch.cuda.Stream(device = self.dev)          # copy stream (CE work + flag writes/waits)
        self.cs3 = [torch.cuda.Stream(device = self.dev) for _ in range(world - 1)]     # one per peer (all-to-all)
        self.ev = [torch.cuda.Event() for _ in range(8)]
        self.nxt, self.prv = (rank + 1) % world, (rank - 1) % world

    @classmethod
    def create(cls, rank, world, slab_bytes, max_cids = 64, rs_rows = 0):
        """collective; None on every rank if the build or IPC setup fails anywhere"""
        ok = torch.ones(1, device = torch.device("cuda", torch.cuda.current_device()))
        try:
            if rank == 0: _load()
        except Exception as e:
            print(f"inkling2_pc: build failed ({e})", flush = True); ok.zero_()
        dist.all_reduce(ok, op = dist.ReduceOp.MIN)
        if ok.item() == 0: return None
        try:
            return cls(rank, world, slab_bytes, max_cids, rs_rows)
        except Exception as e:
            print(f"inkling2_pc: setup failed on rank {rank} ({e})", flush = True)
            return None

    # ---- slab regions (same offsets on every rank: allocate in the same order everywhere)
    def region(self, name, shape, dtype):
        if name in self.regions:
            off, t = self.regions[name]
            assert tuple(t.shape) == tuple(shape) and t.dtype == dtype, f"inkling2_pc: region {name} redefined"
            return t
        n = 1
        for s in shape: n *= s
        nbytes = n * torch.tensor([], dtype = dtype).element_size()
        off = (self.top + 255) // 256 * 256
        assert off + nbytes <= self.cap, f"inkling2_pc: slab full ({name}: {off + nbytes} > {self.cap})"
        t = self.ext.view(self.h, off, list(shape), _DT[dtype])
        self.top = off + nbytes
        self.regions[name] = (off, t)
        return t

    def off_of(self, t):
        return t.data_ptr() - self.ext.base(self.h, self.rank)

    def _cid(self, name):
        if name not in self.cids:
            assert len(self.cids) < self.max_cids
            self.cids[name] = len(self.cids); self.epoch[name] = 0
        return self.cids[name]

    def _flag(self, cid, kind, k):          # kind 0 = data (receiver side), 1 = ack (sender side)
        return (cid * 2 + kind) * NSTEP + k

    # ---- ring all-gather of row blocks inside slab tensors
    def all_gather(self, name, bufs, blocks, consumed_later = False):
        """bufs: slab tensors [.., T, C] (rows on dim -2; several bufs move together, e.g. q/k/v streams); every rank wrote
        its own rows blocks[rank] = (row0, rows); on return (current stream) every block is present on every rank.
        The caller must call release(name) after enqueueing the last consumer of bufs."""
        if not isinstance(bufs, (list, tuple)): bufs = [bufs]
        cid = self._cid(name); self.epoch[name] += 1; e = self.epoch[name]
        cur = torch.cuda.current_stream(self.dev)
        cs = self.cs
        cs.wait_stream(cur)
        W, r, ext, h = self.world, self.rank, self.ext, self.h
        for k in range(W - 1):
            b = (r - k) % W
            if k > 0:
                ext.wait(h, self._flag(cid, 0, k - 1), e, cs.cuda_stream)
            if k == 0 and e > 1:            # r + 1 released the previous epoch's buffer
                ext.wait(h, self._flag(cid, 1, 0), e - 1, cs.cuda_stream)
            r0, nr = blocks[b]
            for t in bufs:
                self._copy_rows(t, r0, nr, cs)
            ext.signal(h, self.nxt, self._flag(cid, 0, k), e, cs.cuda_stream)
        ext.wait(h, self._flag(cid, 0, W - 2), e, cur.cuda_stream)
        cur.wait_stream(cs)

    def release(self, name):
        """after the consumers of an all-gather's buffers: lets r - 1 write the next epoch into them"""
        cid = self.cids[name]
        self.ext.signal(self.h, self.prv, self._flag(cid, 1, 0), self.epoch[name], torch.cuda.current_stream(self.dev).cuda_stream)

    def _copy_rows(self, t, r0, nr, stream):
        """rows [r0, r0 + nr) of slab tensor t ([T, C] or [n, T, C]; rows and columns contiguous, outer dim strided) to
        the same place on rank r + 1"""
        if nr <= 0: return
        C = t.shape[-1]; es = t.element_size()
        assert t.stride(-1) == 1 and t.stride(-2) == C
        outer = t.shape[0] if t.dim() == 3 else 1
        pitch = t.stride(0) * es if t.dim() == 3 else 0
        w = nr * C * es
        src = t.data_ptr() + r0 * C * es
        self.ext.copy(self.h, self.nxt, self.off_of(t) + r0 * C * es, src, w, outer, pitch, pitch, stream.cuda_stream)

    # ---- ring reduce-scatter
    def reduce_scatter(self, name, partial, blocks, out):
        """partial: [T, C] local tensor (fp16/bf16/fp32, contiguous); blocks[o] = (row0, rows), possibly overlapping
        (halo rows); out: [>= blocks[rank].rows, C] receives the sum over ranks of partial[blocks[rank]]."""
        cid = self._cid(name); self.epoch[name] += 1; e = self.epoch[name]
        W, r, ext, h = self.world, self.rank, self.ext, self.h
        T, C = partial.shape
        dt = partial.dtype
        mrows = max(n for _, n in blocks)
        cap = self.rs_rows or mrows
        assert mrows <= cap, f"inkling2_pc: reduce-scatter block of {mrows} rows > slot capacity {cap}"
        par = e & 1
        slots = [self.region(f"rs:{name}:{p}:{k}", (cap, C), dt) for p in range(2) for k in range(W - 1)]
        slot = lambda k: slots[par * (W - 1) + k]
        accs = getattr(self, "_accs", {})
        self._accs = accs
        cur = torch.cuda.current_stream(self.dev)
        cs = self.cs
        cs.wait_stream(cur)                 # partial is ready
        for k in range(W - 1):
            b = (r - 1 - k) % W
            r0, nr = blocks[b]
            if k == 0:
                src = partial[r0:r0 + nr]
            else:
                ext.wait(h, self._flag(cid, 0, k - 1), e, cur.cuda_stream)
                key = (name, k, dt, C)
                if key not in accs or accs[key].shape[0] < mrows:
                    accs[key] = torch.empty((mrows, C), dtype = dt, device = self.dev)
                src = accs[key][:nr]
                ext.sum(src, [slot(k - 1)[:nr], partial[r0:r0 + nr]])
                ext.signal(h, self.prv, self._flag(cid, 1, k - 1), e, cur.cuda_stream)      # consumed slot k-1
                self.ev[k].record(cur)
                cs.wait_event(self.ev[k])
            if e > 2:                        # r + 1 consumed this slot's parity two epochs ago
                ext.wait(h, self._flag(cid, 1, k), e - 2, cs.cuda_stream)
            off = self.off_of(slot(k))
            ext.copy(h, self.nxt, off, src.data_ptr(), nr * C * partial.element_size(), 1, 0, 0, cs.cuda_stream)
            ext.signal(h, self.nxt, self._flag(cid, 0, k), e, cs.cuda_stream)
        r0, nr = blocks[r]
        ext.wait(h, self._flag(cid, 0, W - 2), e, cur.cuda_stream)
        ext.sum(out[:nr], [slot(W - 2)[:nr], partial[r0:r0 + nr]])
        ext.signal(h, self.prv, self._flag(cid, 1, W - 2), e, cur.cuda_stream)
        cur.wait_stream(cs)                 # our sends out of partial / accs are done before the caller reuses them
        return out[:nr]


# ---- compressed expert-input packets (I2_PC_PK=int8 | fp8): one fp16 scale per 32 values; dequantized back to fp16 for
# the W4A16 expert kernel after the gather. int8: symmetric, step = absmax / 127; fp8: e4m3 with absmax -> 448.
import triton
import triton.language as tl

PK_GROUP = 32


@triton.jit
def _pk_quant_kernel(x_ptr, q_ptr, s_ptr, D, NG: tl.constexpr, FP8: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < D
    x = tl.load(x_ptr + t * D + offs, mask = m, other = 0.0).to(tl.float32)
    xg = tl.reshape(x, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(xg), 1)
    if FP8:
        sc = amax / 448.0
    else:
        sc = amax / 127.0
    sc = sc.to(tl.float16).to(tl.float32)
    inv = tl.where(sc > 0, 1.0 / sc, 0.0)
    y = xg * inv[:, None]
    if FP8:
        qv = y.to(tl.float8e4nv)
        tl.store(q_ptr + t * D + offs, tl.reshape(qv, (BLOCK,)).to(tl.uint8, bitcast = True), mask = m)
    else:
        qv = tl.extra.cuda.libdevice.rint(y)
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + t * D + offs, tl.reshape(qv, (BLOCK,)).to(tl.uint8, bitcast = True), mask = m)
    goffs = tl.arange(0, BLOCK // 32)
    tl.store(s_ptr + t * NG + goffs, sc.to(tl.float16), mask = goffs < NG)


@triton.jit
def _pk_dequant_kernel(q_ptr, s_ptr, x_ptr, D, NG: tl.constexpr, FP8: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < D
    qb = tl.load(q_ptr + t * D + offs, mask = m, other = 0)
    if FP8:
        v = qb.to(tl.float8e4nv, bitcast = True).to(tl.float32)
    else:
        v = qb.to(tl.int8, bitcast = True).to(tl.float32)
    goffs = tl.arange(0, BLOCK // 32)
    sc = tl.load(s_ptr + t * NG + goffs, mask = goffs < NG, other = 0.0).to(tl.float32)
    y = tl.reshape(tl.reshape(v, (BLOCK // 32, 32)) * sc[:, None], (BLOCK,))
    tl.store(x_ptr + t * D + offs, y.to(tl.float16), mask = m)


def pk_quant(x, q, s, fp8):
    """x [n, D] fp16 -> q [n, D] uint8 codes, s [n, D / 32] fp16 scales"""
    n, D = x.shape
    BLOCK = triton.next_power_of_2(D)
    _pk_quant_kernel[(n,)](x, q, s, D, D // PK_GROUP, fp8, BLOCK, num_warps = 8)


def pk_dequant(q, s, x, fp8):
    n, D = x.shape
    BLOCK = triton.next_power_of_2(D)
    _pk_dequant_kernel[(n,)](q, s, x, D, D // PK_GROUP, fp8, BLOCK, num_warps = 8)
