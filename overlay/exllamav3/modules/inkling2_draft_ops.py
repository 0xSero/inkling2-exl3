"""
Fused Triton kernels for the DSpark draft (inkling2_draft.py): each replaces a chain of ~10-40 small torch kernels per
layer (the draft step was ~600 launches). Math follows the torch reference in inkling2_draft.py: fp32 internally, one
bf16 rounding where the reference rounds, IEEE division + round-half-even for the int8 KV codes.
- qk_rope_write: q/k RMS norm (per 128-wide head) + NeoX RoPE, q -> q_out, k/v -> draft KV cache (bf16 ring for the
  local layers, int8 + fp16 scale paged for the global layer); rows from (slot, anchor pos) blocks or explicit rows.
- add_rms: res = bf16(res + y); xn = bf16(rms(res) * w)
- silu_mul: bf16(silu(g) * u) over a [R, 2I] gate|up row
- mix_rms: rms(sum_t alpha[l, t] * feats[:, t]) * w[l] for several (alpha row, weight) pairs at once
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rms_rope_head(x1, x2, w_ptr, inv, ang_pos, eps):
    """x1, x2: [64] fp32 halves of one 128-wide head (bf16 values). Returns the roped bf16-rounded halves (fp32)"""
    offs = tl.arange(0, 64)
    ss = tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)
    r = libdevice.rsqrt(ss / 128.0 + eps)
    w1 = tl.load(w_ptr + offs).to(tl.float32)
    w2 = tl.load(w_ptr + 64 + offs).to(tl.float32)
    y1 = (x1 * r * w1).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * r * w2).to(tl.bfloat16).to(tl.float32)
    ang = ang_pos * inv
    c = libdevice.cos(ang)
    s = libdevice.sin(ang)
    return (y1 * c - y2 * s).to(tl.bfloat16), (y2 * c + y1 * s).to(tl.bfloat16)


@triton.jit
def _store_kv(dst_ptr, sc_ptr, row, HKL: tl.constexpr, hk, v1, v2, Q8: tl.constexpr):
    """v1, v2: [64] bf16 halves -> cache row (bf16, or int8 codes + fp16 per-(row, head) scale)"""
    offs = tl.arange(0, 64)
    base = (row * HKL + hk) * 128
    if Q8:
        f1 = v1.to(tl.float32)
        f2 = v2.to(tl.float32)
        am = tl.maximum(tl.max(tl.abs(f1), 0), tl.max(tl.abs(f2), 0))
        sc = tl.maximum(libdevice.div_rn(am, 127.0), 1e-8)
        q1 = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(f1, sc)), -128.0), 127.0)
        q2 = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(f2, sc)), -128.0), 127.0)
        tl.store(dst_ptr + base + offs, q1.to(tl.int8))
        tl.store(dst_ptr + base + 64 + offs, q2.to(tl.int8))
        tl.store(sc_ptr + row * HKL + hk, sc.to(tl.float16))
    else:
        tl.store(dst_ptr + base + offs, v1)
        tl.store(dst_ptr + base + 64 + offs, v2)


@triton.jit
def _qk_rope_write_kernel(x_ptr, sx, q_out_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, pt_stride, slot_ptr, pos_ptr,
                          qn_ptr, kn_ptr, inv_ptr, eps, koff, voff,
                          HL: tl.constexpr, HKL: tl.constexpr, IS_GLOBAL: tl.constexpr, RING: tl.constexpr,
                          PS: tl.constexpr, BLK: tl.constexpr, HAS_Q: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    if BLK > 0:                         # rows are blocks of BLK positions starting at the anchor position of sequence r // BLK
        b = r // BLK
        slot = tl.load(slot_ptr + b)
        pos = tl.load(pos_ptr + b) + r % BLK
    else:
        slot = tl.load(slot_ptr + r)
        pos = tl.load(pos_ptr + r)
    offs = tl.arange(0, 64)
    inv = tl.load(inv_ptr + offs)
    ang_pos = pos.to(tl.float32)
    xrow = x_ptr + r.to(tl.int64) * sx
    NQ: tl.constexpr = HL if HAS_Q else 0
    if h < NQ:
        x1 = tl.load(xrow + h * 128 + offs).to(tl.float32)
        x2 = tl.load(xrow + h * 128 + 64 + offs).to(tl.float32)
        y1, y2 = _rms_rope_head(x1, x2, qn_ptr, inv, ang_pos, eps)
        qo = q_out_ptr + r.to(tl.int64) * (HL * 128) + h * 128
        tl.store(qo + offs, y1)
        tl.store(qo + 64 + offs, y2)
    else:
        hk = h - NQ
        if IS_GLOBAL:
            pg = tl.load(pt_ptr + slot.to(tl.int64) * pt_stride + pos // PS)
            row = pg.to(tl.int64) * PS + pos % PS
        else:
            row = slot.to(tl.int64) * RING + pos % RING
        x1 = tl.load(xrow + koff + hk * 128 + offs).to(tl.float32)
        x2 = tl.load(xrow + koff + hk * 128 + 64 + offs).to(tl.float32)
        k1, k2 = _rms_rope_head(x1, x2, kn_ptr, inv, ang_pos, eps)
        _store_kv(kc_ptr, ks_ptr, row, HKL, hk, k1, k2, IS_GLOBAL)
        v1 = tl.load(xrow + voff + hk * 128 + offs)
        v2 = tl.load(xrow + voff + hk * 128 + 64 + offs)
        _store_kv(vc_ptr, vs_ptr, row, HKL, hk, v1, v2, IS_GLOBAL)


def qk_rope_write(x, koff, voff, q_out, kc, vc, ks, vs, page_table, slot, pos, qn, kn, inv, Hl, Hkl, is_global, ring, ps,
                  blk = 0, eps = 1e-6):
    """x [R, *] bf16 rows holding q heads at 0, k heads at koff, v heads at voff (columns). q_out None: k/v only.
    blk > 0: slot/pos are per sequence [R / blk] (pos = anchor position); else per row [R]"""
    R = x.shape[0]
    has_q = q_out is not None
    grid = (R, (Hl if has_q else 0) + Hkl)
    _qk_rope_write_kernel[grid](x, x.stride(0), q_out if has_q else x, kc, vc, ks if ks is not None else kc,
                                vs if vs is not None else vc, page_table, page_table.stride(0), slot, pos,
                                qn if qn is not None else kn, kn, inv, eps, koff, voff,
                                HL = Hl, HKL = Hkl, IS_GLOBAL = is_global, RING = ring, PS = ps, BLK = blk, HAS_Q = has_q,
                                num_warps = 1)


@triton.jit
def _add_rms_kernel(res_ptr, y_ptr, w_ptr, res_out_ptr, xn_ptr, D, eps, ADD: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BD)
    m = offs < D
    x = tl.load(res_ptr + r * D + offs, mask = m, other = 0.0).to(tl.float32)
    if ADD:
        x = (x + tl.load(y_ptr + r * D + offs, mask = m, other = 0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(res_out_ptr + r * D + offs, x, mask = m)
        x = x.to(tl.float32)
    rs = libdevice.rsqrt(tl.sum(x * x, 0) / D + eps)
    w = tl.load(w_ptr + offs, mask = m, other = 0.0).to(tl.float32)
    tl.store(xn_ptr + r * D + offs, (x * rs * w).to(tl.bfloat16), mask = m)


def add_rms(res, y, w, eps = 1e-6):
    """res, y: [R, D] bf16 (y None: no add). Returns (bf16(res + y), bf16(rms(.) * w))"""
    R, D = res.shape
    xn = torch.empty_like(res)
    out = torch.empty_like(res) if y is not None else res
    _add_rms_kernel[(R,)](res, y if y is not None else res, w, out, xn, D, eps, ADD = y is not None,
                          BD = triton.next_power_of_2(D), num_warps = 8)
    return out, xn


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, BI: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BI + tl.arange(0, BI)
    m = c < I
    g = tl.load(gu_ptr + r * 2 * I + c, mask = m, other = 0.0).to(tl.float32)
    u = tl.load(gu_ptr + r * 2 * I + I + c, mask = m, other = 0.0).to(tl.float32)
    s = libdevice.div_rn(g, 1.0 + libdevice.exp(-g))
    tl.store(out_ptr + r * I + c, (s * u).to(tl.bfloat16), mask = m)


def silu_mul(gu):
    R, I2 = gu.shape
    I = I2 // 2
    out = torch.empty((R, I), dtype = torch.bfloat16, device = gu.device)
    BI = 1024
    _silu_mul_kernel[(R, triton.cdiv(I, BI))](gu, out, I, BI = BI, num_warps = 4)
    return out


@triton.jit
def _mix_rms_kernel(f_ptr, a_ptr, w_ptr, out_ptr, R, D, eps, NT: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    l = tl.program_id(1)
    offs = tl.arange(0, BD)
    m = offs < D
    acc = tl.zeros([BD], dtype = tl.float32)
    for t in tl.static_range(NT):
        a = tl.load(a_ptr + l * NT + t)
        acc += tl.load(f_ptr + (r * NT + t) * D + offs, mask = m, other = 0.0).to(tl.float32) * a
    rs = libdevice.rsqrt(tl.sum(acc * acc, 0) / D + eps)
    w = tl.load(w_ptr + l * D + offs, mask = m, other = 0.0).to(tl.float32)
    tl.store(out_ptr + (l * R + r) * D + offs, (acc * rs * w).to(tl.bfloat16), mask = m)


def mix_rms(feats, alpha, w, eps = 1e-6):
    """feats [R, NT, D] bf16, alpha [L, NT] fp32, w [L, D] -> [L, R, D] bf16"""
    R, NT, D = feats.shape
    L = alpha.shape[0]
    out = torch.empty((L, R, D), dtype = torch.bfloat16, device = feats.device)
    _mix_rms_kernel[(R, L)](feats, alpha, w, out, R, D, eps, NT = NT, BD = triton.next_power_of_2(D), num_warps = 8)
    return out
