"""
Rank-local batched Inkling-2 engine: state (KV pages, SWA rings, conv rings), layer math over a token batch, tensor
parallel attention (heads) + expert parallel MoE. One instance per GPU; ranks synchronize through torch.distributed
(NCCL) all-reduces. See inkling2.py for the reference math and inkling2_ops.py for the kernels.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import inkling2_ops as ops
from ..architecture.inkling2 import Inkling2Model, Inkling2Layer

RING_EXTRA = 64            # SWA ring headroom for speculative tokens beyond the window


class Batch:
    """Token batch descriptor (all int32 device tensors of length T; rows of a sequence contiguous, ascending)"""

    def __init__(self, slot, pos, nprev, multi: bool, nsplit_g: int = 1, nsplit_l: int = 4):
        self.slot, self.pos, self.nprev = slot, pos, nprev
        self.T = slot.shape[0]
        self.multi = multi              # more than one row per sequence (prefill chunk / speculative verify)
        self.nsplit_g, self.nsplit_l = nsplit_g, nsplit_l


class I2Engine:
    def __init__(self, model: Inkling2Model, device, rank: int, world: int, max_slots: int, num_pages: int,
                 page_size: int = 64, bits_g: int = 4, bits_l: int = 8, max_pages_per_seq: int = 8192, embed_on_rank: int = 0):
        self.model, self.cfg = model, model.config.i2
        self.device, self.rank, self.world = device, rank, world
        self.max_slots, self.num_pages, self.ps = max_slots, num_pages, page_size
        self.bits_g, self.bits_l = bits_g, bits_l
        self.layers = model.layers
        self.embed = model.embed
        self.head = model.head
        c = self.cfg
        self.Hl, self.Hkl = c.heads // world, c.kv_heads // world
        self.G = self.Hl // self.Hkl
        self.ring = c.window + RING_EXTRA
        dev = device
        D = c.hidden
        self.blocks = None
        dhp = lambda bits: 64 if bits == 4 else 128
        ct = lambda bits: torch.bfloat16 if bits == 16 else torch.int8 if bits == 8 else torch.uint8
        self.kv = {}
        for L in self.layers:
            i = L.idx
            if c.is_global(i):
                shape = (num_pages, page_size, self.Hkl, dhp(bits_g))
                bits = bits_g
            else:
                shape = (max_slots, self.ring, self.Hkl, dhp(bits_l))
                bits = bits_l
            kq = torch.zeros(shape, dtype = ct(bits), device = dev)
            vq = torch.zeros(shape, dtype = ct(bits), device = dev)
            ks = vs = None
            if bits != 16:
                ks = torch.zeros(shape[:-1], dtype = torch.half, device = dev)
                vs = torch.zeros(shape[:-1], dtype = torch.half, device = dev)
            self.kv[i] = (kq, vq, ks, vs)
        z = lambda ch: torch.zeros((max_slots, ops.RC, ch), dtype = torch.bfloat16, device = dev)
        self.rings = {L.idx: (z(self.Hkl * 128), z(self.Hkl * 128), z(D), z(D)) for L in self.layers}
        self.page_table = torch.zeros((max_slots, max_pages_per_seq), dtype = torch.int32, device = dev)
        self.group = None

    # ---- memory accounting
    def kv_bytes(self):
        tot = 0
        for t in self.kv.values():
            tot += sum(x.numel() * x.element_size() for x in t if x is not None)
        return tot

    # ---- collectives
    def allreduce(self, x):
        if self.world > 1:
            dist.all_reduce(x)
        return x

    # ---- embedding / head
    def embed_tokens(self, ids):
        c = self.cfg
        e = self.embed.weight[ids]
        return ops_rms_norm(e, self.embed.norm, c.eps)

    def head_logits(self, h, blocks):
        """h: [T, D] bf16, blocks: [8, T, D]. Returns this rank's softcapped logits shard [T, V/world] fp32"""
        c = self.cfg
        x = ops.attn_res_norm(blocks, c.layers // c.block_layers, h, self.head.out_query[None], self.head.norm, c.eps)[0]
        z = torch.mm(x, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / c.final_cap) * c.final_cap

    # ---- one decoder layer
    def layer(self, L: Inkling2Layer, h, blocks, B: Batch):
        c = self.cfg
        i, T = L.idx, h.shape[0]
        Hl, Hkl, G = self.Hl, self.Hkl, self.G
        nb = c.num_blocks_before(i)
        boundary = c.is_boundary(i)
        kq, vq, ks, vs = self.kv[i]
        kring, vring, aring, mring = self.rings[i]
        bits = self.bits_g if L.is_global else self.bits_l
        q_out = torch.empty((T, Hl, 128), dtype = torch.bfloat16, device = h.device)
        rel_out = torch.empty((T, Hl, 16), dtype = torch.bfloat16, device = h.device)
        write_ring = not B.multi

        # attention input(s) and projections
        if L.is_global:
            xs = ops.attn_res_norm(blocks, nb, h, L.attn_query, L.qkv_norms, c.eps)        # [3, T, D]
            qb = torch.mm(xs[0], L.wq_cat.t())
            kb = torch.mm(xs[1], L.wk_cat.t())
            vb = torch.mm(xs[2], L.wv_cat.t())
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            ops.qkv_post(qb, kb, vb, L.kc_static, L.vc_static, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                         q_out, rel_out, kq, vq, ks, vs, self.page_table, c, True, bits, rel_off, Hkl * 128, Hkl * 128,
                         self.ring, self.ps, write_ring, Hkl, G)
            kraw, vraw = kb, vb
            Nq = qb
        else:
            if i == 0:
                x = ops.attn_res_norm(blocks, 0, h, torch.zeros((1, c.hidden), device = h.device), L.attn_norm, c.eps)
            else:
                x = ops.attn_res_norm(blocks, nb, h, L.attn_query[None], L.attn_norm, c.eps)
            qb = torch.mm(x[0], L.wqkv.t())
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            koff = sink_off + Hl
            kbv, vbv = qb[:, koff:], qb[:, koff + Hkl * 128:]
            ops.qkv_post(qb, kbv, vbv, L.kc_w, L.vc_w, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                         q_out, rel_out, kq, vq, ks, vs, None, c, False, bits, rel_off, 0, 0, self.ring, self.ps,
                         write_ring, Hkl, G)
            kraw, vraw = kbv, vbv
            Nq = qb
        if B.multi:
            ops.ring_write(kraw[:, :Hkl * 128], kring, B.slot, B.pos)
            ops.ring_write(vraw[:, :Hkl * 128], vring, B.slot, B.pos)
        gate = Nq[:, gate_off:gate_off + Hl * 128]
        sink = Nq[:, sink_off:sink_off + Hl]
        R = torch.mm(rel_out.view(T * Hl, 16).float(), L.rel_proj.float()).view(T, Hl, c.rel_extent)
        o = torch.empty((T, Hl * 128), dtype = torch.bfloat16, device = h.device)
        nsplit = B.nsplit_g if L.is_global else B.nsplit_l
        pm = torch.empty((T, Hl, nsplit), dtype = torch.float32, device = h.device)
        pl = torch.empty((T, Hl, nsplit), dtype = torch.float32, device = h.device)
        po = torch.empty((T, Hl, nsplit, 128), dtype = torch.float32, device = h.device)
        ops.attention(q_out, R, sink, gate, kq, vq, ks, vs, self.page_table if L.is_global else None, B.slot, B.pos,
                      (pm, pl, po), o, c, L.is_global, bits, self.ring, self.ps, nsplit, Hkl, G)
        oa = torch.mm(o, L.wo.t(), out_dtype = torch.float32)
        self.allreduce(oa)

        # post-attention conv + residual, attention residual for the MLP input
        resid = torch.empty((T, c.hidden), dtype = torch.float32, device = h.device)
        ops.sconv(oa, L.attn_sconv, aring, B.slot, B.pos, B.nprev, 0, h = h, resid = resid, boundary = boundary,
                  write_ring = write_ring)
        if B.multi:
            ops.ring_write(oa, aring, B.slot, B.pos)
        if boundary:
            blocks[nb].copy_(h)
        nb_m = nb + 1 if boundary else nb
        xm = ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], L.mlp_norm, c.eps, out_f16 = True)[0]

        # routed experts (expert parallel), summed over ranks
        moe = L.mlp.forward(xm, {})
        moe = moe.view(T, c.hidden)
        self.allreduce(moe)
        h_new = torch.empty_like(h)
        ops.sconv(moe, L.mlp_sconv, mring, B.slot, B.pos, B.nprev, 1, resid = resid, out = h_new, write_ring = write_ring)
        if B.multi:
            ops.ring_write(moe, mring, B.slot, B.pos)
        return h_new

    def new_blocks(self, T):
        c = self.cfg
        return torch.zeros((c.layers // c.block_layers, T, c.hidden), dtype = torch.bfloat16, device = self.device)

    def forward(self, ids, B: Batch, stop_layer = None):
        """ids: [T] token ids (all ranks pass the same ids). Returns this rank's logits shard [T, V/world]"""
        h = self.embed_tokens(ids)
        blocks = self.new_blocks(h.shape[0])
        for L in self.layers:
            h = self.layer(L, h, blocks, B)
            if stop_layer is not None and L.idx == stop_layer:
                return h
        return self.head_logits(h, blocks)


def ops_rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return (w.float() * xf).to(torch.bfloat16)
