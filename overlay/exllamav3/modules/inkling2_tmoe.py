"""
Inkling-2 prefill routed experts: Triton grouped GEMMs with in-kernel EXL3 trellis decode (K4, mcg codebook) and large
row tiles, for the expert-major prefill domain (routes sorted by expert, BM-row tiles per expert).

Why: the b12x fused W4A16 kernel decodes every weight once per 16..64-row route block and tops out at ~100 TFLOP/s at
prefill sizes (decode ALU energy under the 300 W cap); here each decoded weight feeds BM = 128 rows.

Numerics are exllamav3's: x @ W = had128((x * suh) had128 W~) * svh per projection, W~ = mcg-decoded fp16 trellis values
(exact, as ext.reconstruct), fp32 accumulate, fp32 Hadamards / scales / situ activation, fp16 intermediate.

Trellis tile (16 x 16 weights, K = 4: 32 u32 words): lane l of the mma.sync B fragment owns states i = 0..7,
    state_i = ((w[l - 1] << 32 | w[l]) >> (28 - 4 i)) & 0xffff,
holding tile element (k, n) with l = (n % 8) * 4 + (k % 8) // 2 and i = k % 2 + 2 (k // 8) + 4 (n // 8)
(exllamav3 exl3_dq.cuh dq8_aligned_4bits; checked against ext.reconstruct by check_decode()).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

HAD_SCALE = 0.08838834764831845      # 1 / sqrt(128)


@triton.jit
def _mcg(state):
    """16-bit trellis state (uint32) -> fp16 codebook value (exllamav3 decode_3inst<1>)"""
    x = state * 0xCBAC1FED
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return lo + hi


@triton.jit
def _tile_geom(BK: tl.constexpr, BN: tl.constexpr, NT):
    """compact word offsets of a [BK, BN] B tile (one row of 32 u32 words per 16x16 trellis tile, tile-aligned
    origin; the per-K-step offset is (k0 // 16) * NT * 32), plus the in-tile gather indices / window shifts"""
    TN: tl.constexpr = BN // 16
    TKN: tl.constexpr = (BK // 16) * TN
    rt = tl.arange(0, TKN)[:, None]
    rl = tl.arange(0, 32)[None, :]
    woff = (rt // TN) * (NT * 32) + (rt % TN) * 32 + rl              # [TKN, 32]
    e = tl.arange(0, 256)[None, :]                                    # in-tile element, row-major (kk, nn)
    kk = e // 16
    nn = e % 16
    lane = (nn % 8) * 4 + (kk % 8) // 2
    i = kk % 2 + 2 * (kk // 8) + 4 * (nn // 8)
    lane_b = tl.broadcast_to(lane, (TKN, 256))
    lane_a = tl.broadcast_to((lane + 31) % 32, (TKN, 256))
    sh = tl.broadcast_to((28 - 4 * i).to(tl.uint32), (TKN, 256))
    return woff, lane_b, lane_a, sh


@triton.jit
def _decode_b(w_ptr, woff, lane_b, lane_a, sh, BK: tl.constexpr, BN: tl.constexpr):
    """[BK, BN] fp16 rotated-basis weights from the compact trellis words at w_ptr + woff"""
    words = tl.load(w_ptr + woff)                                     # [TKN, 32] u32
    b = tl.gather(words, lane_b, 1)
    a = tl.gather(words, lane_a, 1)
    st = ((b >> sh) | ((a << (31 - sh)) << 1)) & 0xFFFF
    v = _mcg(st)                                                      # [TKN, 256] = (tk, tn, kk, nn)
    v = tl.reshape(v, (BK // 16, BN // 16, 16, 16))
    v = tl.permute(v, (0, 2, 1, 3))
    return tl.reshape(v, (BK, BN))


@triton.jit
def _decode_b2(w_ptr, woff, BK: tl.constexpr, BN: tl.constexpr):
    """gather-free variant: lane-major states from two compact loads (word l and word l-1 of every tile), then one
    permute from B-fragment order (lane = n_lo * 4 + k12, i = n3 * 4 + k3 * 2 + k0) to [BK, BN]"""
    TN: tl.constexpr = BN // 16
    TKN: tl.constexpr = (BK // 16) * TN
    rl = tl.arange(0, 32)[None, :]
    prev = woff - rl + (rl + 31) % 32
    b = tl.load(w_ptr + woff)[:, :, None]                             # [TKN, 32, 1]
    a = tl.load(w_ptr + prev)[:, :, None]
    sh = (28 - 4 * tl.arange(0, 8)).to(tl.uint32)[None, None, :]
    st = ((b >> sh) | ((a << (31 - sh)) << 1)) & 0xFFFF               # [TKN, 32, 8]
    v = _mcg(st)
    v = tl.reshape(v, (BK // 16, TN, 8, 4, 2, 2, 2))                  # tk, tn, n_lo, k12, n3, k3, k0
    v = tl.permute(v, (0, 5, 3, 6, 1, 4, 2))                          # tk, k3, k12, k0, tn, n3, n_lo
    return tl.reshape(v, (BK, BN))


@triton.jit
def _bfly(x, BM: tl.constexpr, HI: tl.constexpr, LO: tl.constexpr):
    y = tl.reshape(x, (BM, HI, 2, LO))
    y = tl.permute(y, (0, 1, 3, 2))
    p, q = tl.split(y)
    y = tl.join(p + q, p - q)
    y = tl.permute(y, (0, 1, 3, 2))
    return tl.reshape(y, (BM, 128))


@triton.jit
def _had128(x, BM: tl.constexpr):
    """natural-order Sylvester H128 (normalized) along the last dim of a [BM, 128] fp32 tile"""
    x = _bfly(x, BM, 64, 1)
    x = _bfly(x, BM, 32, 2)
    x = _bfly(x, BM, 16, 4)
    x = _bfly(x, BM, 8, 8)
    x = _bfly(x, BM, 4, 16)
    x = _bfly(x, BM, 2, 32)
    x = _bfly(x, BM, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _decode_full_kernel(w_ptr, out_ptr, NT, N, BK: tl.constexpr, BN: tl.constexpr, DEC: tl.constexpr = 1):
    pk = tl.program_id(0); pn = tl.program_id(1)
    woff, lb, la, sh = _tile_geom(BK, BN, NT)
    shift = pk * (BK // 16) * NT * 32 + pn * (BN // 16) * 32
    if DEC == 2:
        v = _decode_b2(w_ptr + shift, woff, BK, BN)
    else:
        v = _decode_b(w_ptr + shift, woff, lb, la, sh, BK, BN)
    rk = pk * BK + tl.arange(0, BK)[:, None]; rn = pn * BN + tl.arange(0, BN)[None, :]
    tl.store(out_ptr + rk * N + rn, v)


def decode_full(trellis: torch.Tensor, dec: int = 1) -> torch.Tensor:
    """[K/16, N/16, 64] int16 K4 mcg trellis -> [K, N] fp16 (rotated basis), for checks"""
    KT, NT = trellis.shape[0], trellis.shape[1]
    out = torch.empty((KT * 16, NT * 16), dtype=torch.half, device=trellis.device)
    w = trellis.contiguous().view(torch.int32).view(torch.uint32) if hasattr(torch, "uint32") else trellis.contiguous().view(torch.int32)
    _decode_full_kernel[(KT // 4, NT // 4)](w, out, NT, NT * 16, BK=64, BN=64, DEC=dec)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# per-route input rotation: A_g[r] = had128(x[tok[r]] * suh_g[e[r]]), A_u likewise

@triton.jit
def _rot_in_kernel(x_ptr, tok_ptr, exp_ptr, sg_ptr, su_ptr, ag_ptr, au_ptr, R, H, BR: tl.constexpr, R_DEV: tl.constexpr = False):
    pr = tl.program_id(0); pk = tl.program_id(1)
    if R_DEV:
        R = tl.load(R + 0)
    if pr * BR >= R:
        return
    rr = pr * BR + tl.arange(0, BR)
    m = rr < R
    t = tl.load(tok_ptr + rr, mask=m, other=0)
    e = tl.load(exp_ptr + rr, mask=m, other=0)
    rk = pk * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + t[:, None] * H + rk[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    sg = tl.load(sg_ptr + e[:, None] * H + rk[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    su = tl.load(su_ptr + e[:, None] * H + rk[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    g = _had128(x * sg, BR)
    u = _had128(x * su, BR)
    tl.store(ag_ptr + rr[:, None] * H + rk[None, :], g.to(tl.float16), mask=m[:, None])
    tl.store(au_ptr + rr[:, None] * H + rk[None, :], u.to(tl.float16), mask=m[:, None])


# ---------------------------------------------------------------------------------------------------------------------
# FC1: Z[r, n] = had128(situ(had128(A_g W~g) * svh_g, had128(A_u W~u) * svh_u) * suh_d)[n]   (FC2-rotated, fp16)

@triton.jit
def _softcap(x, c):
    # tanh(x / c) * c via exp (stable for |x| large)
    z = x / c
    e = tl.exp(-2.0 * tl.abs(z))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(z >= 0, t, -t) * c


@triton.jit
def _fc1_kernel(ag_ptr, au_ptr, w_ptr, iv_ptr, z_ptr, t_exp, t_r0, t_rows, E, H: tl.constexpr, I: tl.constexpr,
                BM: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, DEC: tl.constexpr = 1):
    NT: tl.constexpr = I // 16
    pt = tl.program_id(0); pn = tl.program_id(1)
    e = tl.load(t_exp + pt); r0 = tl.load(t_r0 + pt); nr = tl.load(t_rows + pt)
    if nr <= 0:
        return
    BN: tl.constexpr = 128
    rm = tl.arange(0, BM); rk = tl.arange(0, BK)
    mrow = rm < nr
    a_off = (r0 + rm)[:, None] * H + rk[None, :]
    woff, lb, la, sh = _tile_geom(BK, BN, NT)
    KT = H // 16
    wg = w_ptr + e.to(tl.int64) * (KT * NT * 32) + pn * (BN // 16) * 32
    wu = w_ptr + (E + e).to(tl.int64) * (KT * NT * 32) + pn * (BN // 16) * 32
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, H, BK, num_stages=STAGES):
        a_g = tl.load(ag_ptr + a_off + k0, mask=mrow[:, None], other=0.0)
        a_u = tl.load(au_ptr + a_off + k0, mask=mrow[:, None], other=0.0)
        ks = (k0 // 16) * NT * 32
        if DEC == 2:
            b_g = _decode_b2(wg + ks, woff, BK, BN)
            b_u = _decode_b2(wu + ks, woff, BK, BN)
        else:
            b_g = _decode_b(wg + ks, woff, lb, la, sh, BK, BN)
            b_u = _decode_b(wu + ks, woff, lb, la, sh, BK, BN)
        acc_g = tl.dot(a_g, b_g, acc_g)
        acc_u = tl.dot(a_u, b_u, acc_u)
    rn = pn * BN + tl.arange(0, BN)
    ivb = iv_ptr + e.to(tl.int64) * (3 * I)
    g = _had128(acc_g, BM) * tl.load(ivb + rn)[None, :].to(tl.float32)
    u = _had128(acc_u, BM) * tl.load(ivb + I + rn)[None, :].to(tl.float32)
    act = _softcap(g, 4.0) * tl.sigmoid(g) * _softcap(u, 25.0)
    z = _had128(act * tl.load(ivb + 2 * I + rn)[None, :].to(tl.float32), BM)
    tl.store(z_ptr + (r0 + rm)[:, None] * I + rn[None, :], z.to(tl.float16), mask=mrow[:, None])


# FC2: Y[r, n] = had128(Z W~d)[n] * svh_d[n] * w[r]   (fp16 per route)

@triton.jit
def _fc2_kernel(z_ptr, w_ptr, svh_ptr, rw_ptr, y_ptr, t_exp, t_r0, t_rows, H: tl.constexpr, I: tl.constexpr,
                BM: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, DEC: tl.constexpr = 1):
    NT: tl.constexpr = H // 16
    pt = tl.program_id(0); pn = tl.program_id(1)
    e = tl.load(t_exp + pt); r0 = tl.load(t_r0 + pt); nr = tl.load(t_rows + pt)
    if nr <= 0:
        return
    BN: tl.constexpr = 128
    rm = tl.arange(0, BM); rk = tl.arange(0, BK)
    mrow = rm < nr
    a_off = (r0 + rm)[:, None] * I + rk[None, :]
    woff, lb, la, sh = _tile_geom(BK, BN, NT)
    KT = I // 16
    wd = w_ptr + e.to(tl.int64) * (KT * NT * 32) + pn * (BN // 16) * 32
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, I, BK, num_stages=STAGES):
        a = tl.load(z_ptr + a_off + k0, mask=mrow[:, None], other=0.0)
        ks = (k0 // 16) * NT * 32
        if DEC == 2:
            b = _decode_b2(wd + ks, woff, BK, BN)
        else:
            b = _decode_b(wd + ks, woff, lb, la, sh, BK, BN)
        acc = tl.dot(a, b, acc)
    rn = pn * BN + tl.arange(0, BN)
    rw = tl.load(rw_ptr + r0 + rm, mask=mrow, other=0.0)
    y = _had128(acc, BM) * tl.load(svh_ptr + e.to(tl.int64) * H + rn)[None, :].to(tl.float32) * rw[:, None]
    tl.store(y_ptr + (r0 + rm)[:, None] * H + rn[None, :], y.to(tl.float16), mask=mrow[:, None])


def _u32(t):
    return t.view(torch.int32).view(torch.uint32)


class TritonExperts:
    """one layer slice's experts: w13 [2, E, H/16, I/16, 64] int16, w2 [E, I/16, H/16, 64] int16 (b12x storage),
    scales (suh_g [E, H], suh_u [E, H], iv [E, 3 I], svh_d [E, H])"""

    def __init__(self, w13, w2, scales, H, I, E, BM=128, BK=64, stages=3, warps=8):
        self.w13, self.w2 = _u32(w13.contiguous()), _u32(w2.contiguous())
        self.suh_g, self.suh_u, self.iv, self.svh_d = [s.contiguous() for s in scales]
        self.H, self.I, self.E = H, I, E
        self.BM, self.BK, self.stages, self.warps = BM, BK, stages, warps

    def plan(self, ids, lo, hi, experts=None, BM=None):
        """routes of local experts (optionally only `experts`), sorted by expert; host-side tile list (prototype)"""
        BM = BM or self.BM
        T, K = ids.shape
        flat = ids.reshape(-1).long()
        m = (flat >= lo) & (flat < hi)
        if experts is not None:
            sel = torch.zeros(hi - lo, dtype=torch.bool, device=ids.device); sel[experts] = True
            m &= sel[(flat - lo).clamp(0, hi - lo - 1)]
        r = m.nonzero().squeeze(1)
        e = flat[r] - lo
        order = torch.argsort(e, stable=True)
        r, e = r[order], e[order]
        cnt = torch.bincount(e, minlength=hi - lo).cpu().tolist()
        te, tr0, tn = [], [], []
        off = 0
        for ex, c in enumerate(cnt):
            for s in range(0, c, BM):
                te.append(ex); tr0.append(off + s); tn.append(min(BM, c - s))
            off += c
        dev = ids.device
        mk = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
        return dict(route=r, tok=(r // K).int(), exp=e.int(), te=mk(te), tr0=mk(tr0), tn=mk(tn), R=int(r.numel()), BM=BM)

    def run(self, x, w_route_flat, P, out=None):
        """x [T, H] fp16; w_route_flat [T*K] fp32; returns out [T, H] fp32 += routed sum of the planned routes"""
        H, I, R = self.H, self.I, P["R"]
        dev = x.device
        if out is None:
            out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=dev)
        if R == 0:
            return out
        ag = torch.empty((R, H), dtype=torch.half, device=dev); au = torch.empty_like(ag)
        _rot_in_kernel[(triton.cdiv(R, 32), H // 128)](x, P["tok"], P["exp"], self.suh_g, self.suh_u, ag, au, R, H, BR=32, num_warps=4)
        z = torch.empty((R, I), dtype=torch.half, device=dev)
        nt = P["te"].numel(); BM = P["BM"]
        _fc1_kernel[(nt, I // 128)](ag, au, self.w13, self.iv, z, P["te"], P["tr0"], P["tn"], self.E, H=H, I=I,
                                    BM=BM, BK=self.BK, STAGES=self.stages, num_warps=self.warps)
        y = torch.empty((R, H), dtype=torch.half, device=dev)
        rw = w_route_flat[P["route"]].float().contiguous()
        svh = self.svh_d
        _fc2_kernel[(nt, H // 128)](z, self.w2, svh, rw, y, P["te"], P["tr0"], P["tn"], H=H, I=I,
                                    BM=BM, BK=self.BK, STAGES=self.stages, num_warps=self.warps)
        out.index_add_(0, P["tok"].long(), y.float())
        return out


# ---------------------------------------------------------------------------------------------------------------------
# graph-safe engine path: device-side planning (no host sync), preallocated workspaces per batch size, deterministic
# per-token sum of the routed outputs in top-k slot order

@triton.jit
def _sum_kernel(y_ptr, pos_ptr, base_ptr, out_ptr, H, base_stride, K: tl.constexpr, BH: tl.constexpr, HAS_BASE: tl.constexpr,
                OUT_F32: tl.constexpr):
    t = tl.program_id(0); ph = tl.program_id(1)
    rh = ph * BH + tl.arange(0, BH)
    acc = tl.zeros((BH,), dtype=tl.float32)
    if HAS_BASE:
        acc += tl.load(base_ptr + t.to(tl.int64) * base_stride + rh).to(tl.float32)
    for k in tl.static_range(K):
        p = tl.load(pos_ptr + t * K + k)
        if p >= 0:
            acc += tl.load(y_ptr + p.to(tl.int64) * H + rh).to(tl.float32)
    if OUT_F32:
        tl.store(out_ptr + t.to(tl.int64) * H + rh, acc)
    else:
        tl.store(out_ptr + t.to(tl.int64) * H + rh, acc.to(tl.float16))


class TritonMoE:
    """hybrid prefill experts for one rank: experts with >= hot_rows routed rows in this batch run here (BM-row tiles,
    decode amortized over BM rows), the rest stay on the caller's b12x path (their routes are remapped to a remote id).
    All planning is on device, so the call is CUDA-graph safe and never syncs the host."""

    def __init__(self, layers, H, I, E, lo, hi, topk, dev, BM=128, BK=64, stages=3, warps=8, hot_rows=128,
                 max_route_frac=0.75):
        self.layers = {li: (_u32(L.w13.contiguous()), _u32(L.w2.contiguous()), [s.contiguous() for s in L.scales])
                       for li, L in layers.items()}
        self.H, self.I, self.E, self.lo, self.hi, self.K = H, I, E, lo, hi, topk
        self.BM, self.BK, self.stages, self.warps, self.hot_rows = BM, BK, stages, warps, hot_rows
        self.max_route_frac = max_route_frac
        self.dev = dev
        self.ws = {}

    def _ws(self, T):
        if T not in self.ws:
            H, I, K, dev = self.H, self.I, self.K, self.dev
            RM = max(self.BM, int(T * K * self.max_route_frac))
            MT = RM // self.BM + self.E
            self.ws[T] = dict(RM=RM, MT=MT, ag=torch.empty((RM, H), dtype=torch.half, device=dev),
                              au=torch.empty((RM, H), dtype=torch.half, device=dev),
                              z=torch.empty((RM, I), dtype=torch.half, device=dev),
                              y=torch.empty((RM, H), dtype=torch.half, device=dev),
                              ar=torch.arange(MT, dtype=torch.int64, device=dev),
                              ones=torch.ones(T * K, dtype=torch.int32, device=dev))
        return self.ws[T]

    def plan(self, ids, w, T):
        """ids [T, K] int32 global expert ids, w [T, K] fp32 -> device plan (hot routes sorted by expert, tiles)"""
        W = self._ws(T); E, K, BM, RM, MT = self.E, self.K, self.BM, W["RM"], W["MT"]
        flat = ids.reshape(-1).long()
        loc = (flat >= self.lo) & (flat < self.hi)
        e = torch.where(loc, flat - self.lo, torch.full_like(flat, E))
        cnt = torch.zeros(E + 1, dtype=torch.int32, device=ids.device).scatter_add_(0, e, W["ones"][:flat.numel()])[:E]
        # hot experts: >= hot_rows rows, largest first while the routes fit the workspace
        cs, order = torch.sort(cnt, descending=True)
        fit = (torch.cumsum(cs, 0) <= RM) & (cs >= self.hot_rows)
        hot = torch.zeros(E + 1, dtype=torch.bool, device=ids.device)
        hot[:E].scatter_(0, order, fit)
        sel = hot[e]                                                     # hot local route (e == E -> False)
        key = torch.where(sel, e, torch.full_like(e, E))
        skey, idx = torch.sort(key, stable=True)                         # hot routes first, by expert
        idx = idx[:RM]; skey = skey[:RM]
        cnt_h = torch.where(hot[:E], cnt, torch.zeros_like(cnt)).long()
        seg = torch.cumsum(cnt_h, 0) - cnt_h
        nt = (cnt_h + BM - 1) // BM
        tend = torch.cumsum(nt, 0)
        t = W["ar"]
        te = torch.searchsorted(tend, t, right=True).clamp(max=E - 1)
        j = t - (tend[te] - nt[te])
        rows = (cnt_h[te] - j * BM).clamp(0, BM)
        rows = torch.where(t < tend[-1], rows, torch.zeros_like(rows))
        R = cnt_h.sum().to(torch.int32).reshape(1)
        valid = torch.arange(RM, device=ids.device) < R
        n = flat.numel()
        pos = torch.full((n + 1,), -1, dtype=torch.int32, device=ids.device)      # slot n: dummy target of invalid rows
        pos.scatter_(0, torch.where(valid, idx, torch.full_like(idx, n)), torch.arange(RM, dtype=torch.int32, device=ids.device))
        pos = torch.where(sel, pos[:n], torch.full_like(pos[:n], -1))
        return dict(tok=(idx // K).to(torch.int32), exp=skey.clamp(max=E - 1).to(torch.int32), rw=w.reshape(-1)[idx].float(),
                    te=te.to(torch.int32), tr0=(seg[te] + j * BM).to(torch.int32), tn=rows.to(torch.int32), R=R, pos=pos,
                    hot_global=hot[:E])

    def run(self, layer_idx, x, P, T, base=None, out_dtype=torch.float32):
        """x [T, H] fp16 -> [T, H] (base + routed sum over the planned hot routes)"""
        W = self._ws(T); H, I = self.H, self.I
        w13, w2, (suh_g, suh_u, iv, svh_d) = self.layers[layer_idx]
        RM, MT = W["RM"], W["MT"]
        _rot_in_kernel[(triton.cdiv(RM, 32), H // 128)](x, P["tok"], P["exp"], suh_g, suh_u, W["ag"], W["au"], P["R"], H,
                                                       BR=32, R_DEV=True, num_warps=4)
        _fc1_kernel[(MT, I // 128)](W["ag"], W["au"], w13, iv, W["z"], P["te"], P["tr0"], P["tn"], self.E, H=H, I=I,
                                    BM=self.BM, BK=self.BK, STAGES=self.stages, num_warps=self.warps)
        _fc2_kernel[(MT, H // 128)](W["z"], w2, svh_d, P["rw"], W["y"], P["te"], P["tr0"], P["tn"], H=H, I=I,
                                    BM=self.BM, BK=self.BK, STAGES=self.stages, num_warps=self.warps)
        out = torch.empty((T, H), dtype=out_dtype, device=x.device)
        assert base is None or base.stride(1) == 1
        _sum_kernel[(T, H // 256)](W["y"], P["pos"], base if base is not None else out, out, H,
                                   base.stride(0) if base is not None else H, K=self.K, BH=256,
                                   HAS_BASE=base is not None, OUT_F32=out_dtype == torch.float32, num_warps=4)
        return out
