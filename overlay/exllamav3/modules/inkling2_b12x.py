"""
Inkling-2 routed experts through the b12x fused Trellis W4A16 MoE kernels (Local Inference Lab, Apache-2.0; now
FlashInfer flashinfer/experimental/b12x, imported as `b12x`), for uniform-rate mcg EXL3 experts (v3: K4).

In-memory adapter, no requantization and no second copy of the weights: per layer the rank's local experts are
repacked into the b12x native trellis_t256 storage (w13 [2, E, H/16, I/16, 16K] projection-major, w2 [E, I/16, H/16,
16K], int16 words byte-identical to exllamav3's per-expert `trellis` tiles) and exllamav3's per-expert tensors become
views into it (the model must be loaded with EXL3_LOAD_ARENA=0 so the old per-expert allocations are really freed),
so the original exl3_moe_coop / wide paths keep working on the same storage (fallback for prefill-sized
batches). The fp16 rotation vectors are repacked the same way: gate/up suh -> [E, H], [gate svh | up svh | down suh]
-> [E, 3I] (b12x's intermediate_rotations order), down svh -> [E, H].

Kernel contract used (validated against fp32 truth on v3 layer 5, rel err 8.6e-4): activation "situ" (beta 4, linear
beta 25 = Inkling's softcap(g,4)*sigmoid(g)*softcap(u,25)), full_rotation + intermediate_rotation, fp16 I/O and
rotation input, expert_map for the EP slice (global id -> local id or -1), router weights fp32 applied after FC2.
"""
from __future__ import annotations
import os, sys
import torch

_FI = os.environ.get("I2_B12X_PATH", "/opt/flashinfer")


def _import_b12x():
    for p in (_FI, _FI + "/flashinfer/experimental/b12x/_compat"):
        if p not in sys.path:
            sys.path.append(p)
    from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_fused_moe, compile_w4a16_topk_sum, run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    return make_w4a16_packed_buffers, compile_w4a16_fused_moe, compile_w4a16_topk_sum, run_w4a16_moe, prepare_trellis256_moe_weights


# (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) per verify row count; override with I2_B12X_TILES="T:a,b,c,d;..."
DEFAULT_TILE = (128, 128, 32, 512)
DEFAULT_TILES = {}


def _tiles_from_env():
    t = dict(DEFAULT_TILES)
    s = os.environ.get("I2_B12X_TILES", "")
    for part in filter(None, s.split(";")):
        k, v = part.split(":")
        t[int(k)] = tuple(int(x) for x in v.split(","))
    return t


def _rebuild_linear_bc(inner, ext, g_tensor_cache):
    inner.bc = ext.BC_LinearEXL3(inner.trellis, inner.suh, inner.svh, inner.K, inner.bias, inner.mcg, inner.mul1,
                                 g_tensor_cache.get(*inner.bsz1_xh_args))


class B12xLayer:
    """Repacked storage + prepared b12x weights of one layer's local experts"""

    def __init__(self, mlp, prep, block_m, tile):
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache
        dev = mlp.device
        E = mlp.num_local_experts
        gates, ups, downs = [l.inner for l in mlp.gates], [l.inner for l in mlp.ups], [l.inner for l in mlp.downs]
        assert all(x.mcg and not x.mul1 for x in gates + ups + downs), "b12x path needs mcg experts"
        K = gates[0].K
        assert all(x.K == K for x in gates + ups + downs), "b12x path needs a uniform bitrate"
        assert all(x.bias is None for x in gates + ups + downs)
        H, I = gates[0].in_features, gates[0].out_features
        self.H, self.I, self.E, self.K = H, I, E, K
        w13 = torch.empty((2, E) + tuple(gates[0].trellis.shape), dtype=torch.int16, device=dev)
        w2 = torch.empty((E,) + tuple(downs[0].trellis.shape), dtype=torch.int16, device=dev)
        suh_g = torch.empty((E, H), dtype=torch.half, device=dev)
        suh_u = torch.empty((E, H), dtype=torch.half, device=dev)
        iv = torch.empty((E, 3, I), dtype=torch.half, device=dev)
        svh_d = torch.empty((E, H), dtype=torch.half, device=dev)
        for e in range(E):
            g, u, d = gates[e], ups[e], downs[e]
            w13[0, e].copy_(g.trellis); g.trellis = w13[0, e]
            w13[1, e].copy_(u.trellis); u.trellis = w13[1, e]
            w2[e].copy_(d.trellis); d.trellis = w2[e]
            suh_g[e].copy_(g.suh); g.suh = suh_g[e]
            suh_u[e].copy_(u.suh); u.suh = suh_u[e]
            iv[e, 0].copy_(g.svh); g.svh = iv[e, 0]
            iv[e, 1].copy_(u.svh); u.svh = iv[e, 1]
            iv[e, 2].copy_(d.suh); d.suh = iv[e, 2]
            svh_d[e].copy_(d.svh); d.svh = svh_d[e]
            for x in (g, u, d):
                _rebuild_linear_bc(x, ext, g_tensor_cache)
        # Rebuild exllamav3's fused-path state on the new views: BC_BlockSparseMLP holds the per-expert
        # BC_LinearEXL3 objects and its own gate/up pointer tables, which would otherwise pin the old copies
        mlp.bc = None
        mlp.multi_gate = mlp.multi_up = mlp.multi_down = None
        if getattr(mlp, "batch_recon", None) not in (None, False):
            mlp.batch_recon = None          # rebuilt lazily from the (new) per-expert tensors
        mlp.load_local()
        self.w13, self.w2 = w13, w2
        self.tier = prep(w13, w2, hidden_size=H, intermediate_size=I, num_experts=E, activation="situ",
                         fc1_tile_n=tile[1], fc2_tile_n=tile[3], params_dtype=torch.float16,
                         w13_layout="trellis_t256_proj", codebook="mcg", gate_suh=suh_g, up_suh=suh_u,
                         intermediate_rotations=iv.view(E, 3 * I), down_svh=svh_d, tile_config=tuple(tile),
                         workspace=B12xMoE.workspace(dev))


class B12xMoE:
    """Engine hook: routed-expert sum of the rank's EP slice for a token batch of one of the prepared sizes"""

    _ws = {}

    @classmethod
    def workspace(cls, dev):
        if dev not in cls._ws:
            from b12x.moe._shared.kernels.w4a16.prepare import _make_workspace
            cls._ws[dev] = _make_workspace(dev, max_blocks_per_sm=4)
        return cls._ws[dev]

    def __init__(self, layers, dev, sizes=(8, 16, 32, 64, 128), block_m=8, tile=None):
        (self.make_buffers, self.compile_moe, self.compile_sum, self.run, prep) = _import_b12x()
        self.dev = dev
        self.block_m = int(os.environ.get("I2_B12X_BLOCK_M", block_m))
        tiles = _tiles_from_env()
        self.tile = tuple(tile or tiles.get(0, DEFAULT_TILE))
        self.tiles = {T: tuple(tiles.get(T, self.tile)) for T in sizes}
        # one prepared weight object per (layer, fc tile N pair); the tile N pair is a preparation contract
        self.layers = {}
        mlp0 = layers[0].mlp
        self.E_all = mlp0.num_experts
        self.lo, self.hi = mlp0.routing_first, mlp0.routing_first + mlp0.num_local_experts
        assert os.environ.get("EXL3_LOAD_ARENA") == "0", \
            "load the model with EXL3_LOAD_ARENA=0: arena-sliced expert tensors are never freed by the repack (2x weights)"
        for L in layers:
            self.layers[L.idx] = B12xLayer(L.mlp, prep, self.block_m, self.tile)
            torch.cuda.empty_cache()        # return the layer's old per-expert segments before the next layer's stack
        lay = next(iter(self.layers.values()))
        self.H, self.I, self.E, self.K = lay.H, lay.I, lay.E, lay.K
        self.topk = mlp0.num_experts_per_tok
        self.expert_map = torch.full((self.E_all,), -1, dtype=torch.int32, device=dev)
        self.expert_map[self.lo:self.hi] = torch.arange(self.E, dtype=torch.int32, device=dev)
        props = torch.cuda.get_device_properties(dev)
        self.sizes = {}
        for T in sizes:
            tile_T = self.tiles[T]
            if tile_T[1] != self.tile[1] or tile_T[3] != self.tile[3]:
                raise ValueError(f"tile N dims for T={T} {tile_T} differ from the prepared tier {self.tile}")
            buf = self.make_buffers(lay.tier, m=T, topk=self.topk, dtype=torch.float16, device=dev,
                                    route_num_experts=self.E_all, full_rotation=True, block_size_m=self.block_m)
            launch = self.compile_moe(size_m=T, hidden_size=self.H, intermediate_size=self.I, num_experts=self.E,
                top_k=self.topk, activation="situ", apply_router_weight_on_input=False, zero_fc2_output=False,
                moe_block_size=self.block_m, max_m_blocks=int(buf.block_expert_ids.numel()), element_dtype="fp16",
                sms=int(props.multi_processor_count), max_shared_mem=int(props.shared_memory_per_block_optin),
                weight_layout="trellis_t256", scale_format="e4m3_k32", w13_layout="trellis_t256_proj",
                trellis_bits=int(self.K), trellis_codebook="mcg", force_tile_config=tile_T, intermediate_rotation=True,
                full_rotation=True, rotation_input_dtype="fp16")
            tsum = self.compile_sum(m=T, topk=self.topk, hidden_size=self.H, element_dtype="fp16", full_rotation=True,
                                    num_experts=self.E, route_num_experts=self.E_all, route_ids_dtype=torch.int32,
                                    use_expert_map=True)
            self.sizes[T] = (buf, launch, tsum)

    def supports(self, T):
        return T in self.sizes

    def forward(self, layer_idx, mlp, x):
        """x: [T, H] fp16 (the MoE input). Returns [T, H] fp32 routed sum over this rank's experts"""
        T = x.shape[0]
        buf, launch, tsum = self.sizes[T]
        lay = self.layers[layer_idx]
        sel, rw = mlp.routing_fn(T, mlp.routing_cfg, x, {})
        ids = sel.to(torch.int32)
        w = rw.float()
        out = self.run(x, lay.tier, w, ids, activation="situ", intermediate_cache13=buf.intermediate_cache13,
            intermediate_cache2=buf.intermediate_cache2, output=buf.output, fc1_c_tmp=buf.fc1_c_tmp,
            fc2_c_tmp=buf.fc2_c_tmp, packed_route_indices=buf.packed_route_indices,
            block_expert_ids=buf.block_expert_ids, packed_route_count=buf.packed_route_count,
            expert_offsets=buf.expert_offsets, expert_counts=buf.expert_counts, expert_map=self.expert_map,
            output_expert_map=self.expert_map, fused_launch=launch, topk_sum_launch=tsum,
            route_block_size_m=self.block_m, intermediate_rotation_scales=lay.tier.intermediate_rotations,
            full_rotation=True, suh_gate_table=lay.tier.gate_suh, suh_up_table=lay.tier.up_suh,
            svh_table=lay.tier.down_svh, rotation_a_gate=buf.rotation_a_gate, rotation_a_up=buf.rotation_a_up)
        y = out if out is not None else buf.output
        return y.view(T, -1)[:, :self.H].float()
