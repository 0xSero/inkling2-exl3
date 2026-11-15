# CUDA source of the SM120 prefill attention kernels (inkling2_pattn.py). Shipped as a Python string so that a pip
# install keeps it (non-Python package files are dropped by the Studio image build).
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

// ---------------------------------------------------------------------------------------------------------------------
// Prefill attention for Inkling-2 per TP rank: 6 query heads per KV head (packed flat as (row, head) MMA rows), head dim 128,
// rank-16 relative bias (bf16 rel_q . bf16 rel_proj, exact products on bf16 tensor cores), per-row sinks, sigmoid gate.
// KIND 0: global layer, 64-row pages of 4-bit q16 codes (code - 7.5) * s, 4 fp16 scales of 32 per 128-vector, Hadamard
//         rotated cache space; the output is rotated back with an fp16 MMA against the (dim-permuted) rotation.
// KIND 1: sliding-window layer (1024), int8 ring, one fp16 scale per (row, head).
// One CTA (8 warps, 16 MMA rows each) streams a contiguous segment of the (unit, key block) work list (stream-K); a unit
// is (tile of 128 flat rows of one sequence, kv head). Units cut by a segment boundary leave partials for the merge kernel.
// K/V tiles (64 keys) are dequantized once per CTA into swizzled fp16 shared memory (double buffered, register prefetch).
// ---------------------------------------------------------------------------------------------------------------------

#define DH 128
#define GQ 6
#define BM 128
#define BN 64
#define NWARP 8
#define NTHR 256
#define EXTENT 1024
#define WINDOW 1024
#define WS 44          // per-warp bias staging row stride (floats)

struct Unit { int t_seq, pos_seq, f0, nf, hk, b_lo, b_hi, slot; };

struct Params {
    const __nv_bfloat16* q; const __nv_bfloat16* relq; const __nv_bfloat16* relpT;
    const __nv_bfloat16* sink; int sink_stride; const __nv_bfloat16* gate; int gate_stride;
    const uint8_t* kq; const uint8_t* vq; const __half* ks; const __half* vs;
    long long cs_pos, cs_h, sc_pos;          // cache strides (elements of kq / ks)
    const int* page_table; int pt_stride; int ring;
    const __half* rot;                       // [128][128] fp16, rot[j][slot] = M[j][perm(slot)] (global only)
    const Unit* units; const int* cum; int n_units; int seg; int total;
    __nv_bfloat16* out;                      // [T, HL * 128]
    float* part_o; float* part_ml;           // [nCTA, 2, BM, 128], [nCTA, 2, BM, 2]
    int HL;
    float qk_scale;                          // sm_scale * log2(e)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
// swizzled [64 keys][16 chunks of 8 fp16] tile: byte offset of (key, chunk)
__device__ __forceinline__ uint32_t swz(int key, int chunk) { return (uint32_t)(key * 256 + ((chunk ^ (key & 7)) << 4)); }

// raw K/V codes of one 64-key block held in registers by each thread (prefetch stage)
template <int KIND> struct Raw;
template <> struct Raw<0> { uint4 k, v; __half ksc, vsc; bool ok; };
template <> struct Raw<1> { uint4 k0, k1, v0, v1; __half ksc, vsc; bool ok; };

template <int KIND>
__device__ __forceinline__ void load_raw(Raw<KIND>& r, const Params& P, const Unit& u, int b, int jlo, int jhi) {
    const int tid = threadIdx.x;
    const int key = tid >> 2, part = tid & 3;
    const int j = b * BN + key;
    r.ok = (j >= jlo) && (j <= jhi);
    long long row;
    if (KIND == 0) {
        int page = __ldg(P.page_table + (long long)u.slot * P.pt_stride + b);
        row = (long long)page * BN + key;
    } else {
        row = (long long)u.slot * P.ring + (j % P.ring);
    }
    if (!r.ok) row = (KIND == 0) ? row : (long long)u.slot * P.ring;   // keep the address in bounds
    const long long base = row * P.cs_pos + (long long)u.hk * P.cs_h;
    if constexpr (KIND == 0) {
        r.k = __ldg(reinterpret_cast<const uint4*>(P.kq + base + part * 16));
        r.v = __ldg(reinterpret_cast<const uint4*>(P.vq + base + part * 16));
        r.ksc = P.ks[row * P.sc_pos + u.hk * 4 + part];
        r.vsc = P.vs[row * P.sc_pos + u.hk * 4 + part];
    } else {
        const uint4* kp = reinterpret_cast<const uint4*>(P.kq + base + part * 32);
        const uint4* vp = reinterpret_cast<const uint4*>(P.vq + base + part * 32);
        r.k0 = __ldg(kp); r.k1 = __ldg(kp + 1); r.v0 = __ldg(vp); r.v1 = __ldg(vp + 1);
        r.ksc = P.ks[row * P.sc_pos + u.hk];
        r.vsc = P.vs[row * P.sc_pos + u.hk];
    }
}

// 4-bit word (8 codes, low nibble = even element) -> 4 half2 in slot order (e0,e4)(e1,e5)(e2,e6)(e3,e7), values (c - 7.5) * s
__device__ __forceinline__ void deq4(uint32_t w, __half2 s, __half2 hs, uint32_t* o) {
    const __half2 m1032 = __floats2half2_rn(1032.f, 1032.f);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t x = ((w >> (4 * i)) & 0x000F000Fu) | 0x64006400u;
        o[i] = h2u(__hfma2(__hsub2(u2h(x), m1032), s, hs));
    }
}
// int8 word (4 codes) -> 2 half2 (e0,e1)(e2,e3), values c * s
__device__ __forceinline__ void deq8(uint32_t w, __half2 s, uint32_t* o) {
    const __half2 m1152 = __floats2half2_rn(1152.f, 1152.f);
    uint32_t u = w ^ 0x80808080u;
    o[0] = h2u(__hmul2(__hsub2(u2h(__byte_perm(u, 0x64646464u, 0x5140)), m1152), s));
    o[1] = h2u(__hmul2(__hsub2(u2h(__byte_perm(u, 0x64646464u, 0x7362)), m1152), s));
}

template <int KIND>
__device__ __forceinline__ void store_tile(const Raw<KIND>& r, uint8_t* Ks, uint8_t* Vs) {
    const int tid = threadIdx.x;
    const int key = tid >> 2, part = tid & 3;
    if constexpr (KIND == 0) {
        __half2 ks2 = __half2half2(r.ksc), vs2 = __half2half2(r.vsc);
        __half2 khs = __hmul2(ks2, __floats2half2_rn(0.5f, 0.5f)), vhs = __hmul2(vs2, __floats2half2_rn(0.5f, 0.5f));
        const uint32_t kw[4] = {r.k.x, r.k.y, r.k.z, r.k.w}, vw[4] = {r.v.x, r.v.y, r.v.z, r.v.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint4 ok, ov;
            uint32_t a[4], c[4];
            deq4(kw[i], ks2, khs, a); deq4(vw[i], vs2, vhs, c);
            ok = r.ok ? make_uint4(a[0], a[1], a[2], a[3]) : make_uint4(0, 0, 0, 0);
            ov = r.ok ? make_uint4(c[0], c[1], c[2], c[3]) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(Ks + swz(key, part * 4 + i)) = ok;
            *reinterpret_cast<uint4*>(Vs + swz(key, part * 4 + i)) = ov;
        }
    } else {
        __half2 ks2 = __half2half2(r.ksc), vs2 = __half2half2(r.vsc);
        const uint32_t kw[8] = {r.k0.x, r.k0.y, r.k0.z, r.k0.w, r.k1.x, r.k1.y, r.k1.z, r.k1.w};
        const uint32_t vw[8] = {r.v0.x, r.v0.y, r.v0.z, r.v0.w, r.v1.x, r.v1.y, r.v1.z, r.v1.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t a[4], c[4];
            deq8(kw[2 * i], ks2, a); deq8(kw[2 * i + 1], ks2, a + 2);
            deq8(vw[2 * i], vs2, c); deq8(vw[2 * i + 1], vs2, c + 2);
            uint4 ok = r.ok ? make_uint4(a[0], a[1], a[2], a[3]) : make_uint4(0, 0, 0, 0);
            uint4 ov = r.ok ? make_uint4(c[0], c[1], c[2], c[3]) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(Ks + swz(key, part * 4 + i)) = ok;
            *reinterpret_cast<uint4*>(Vs + swz(key, part * 4 + i)) = ov;
        }
    }
}

__device__ __forceinline__ int find_unit(const int* cum, int n, int w) {
    int lo = 0, hi = n - 1;                 // largest u with cum[u] <= w
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (__ldg(cum + mid) <= w) lo = mid; else hi = mid - 1; }
    return lo;
}

template <int KIND>
__global__ void __launch_bounds__(NTHR, 1) pattn_kernel(const Params P) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Kbuf[2] = {smem, smem + 2 * 16384};
    uint8_t* Vbuf[2] = {smem + 16384, smem + 3 * 16384};
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    float* Wsm = reinterpret_cast<float*>(smem + 4 * 16384) + warp * 16 * WS;

    const int c = blockIdx.x;
    const int w_lo = c * P.seg;
    const int w_hi = min(P.total, w_lo + P.seg);
    if (w_lo >= w_hi) return;
    int u_idx = find_unit(P.cum, P.n_units, w_lo);
    const int u_first = u_idx;

    for (int w = w_lo; w < w_hi; u_idx++) {
        const Unit u = P.units[u_idx];
        const int ucum = __ldg(P.cum + u_idx), unext = __ldg(P.cum + u_idx + 1);
        const int i0 = w - ucum, i1 = min(w_hi, unext) - ucum;      // local block range of this piece
        w = ucum + i1;
        const bool whole = (i0 == 0) && (i1 == unext - ucum);
        // rows of this warp: flat f = u.f0 + 16 * warp + i (i < 16); thread rows i = gid, gid + 8
        const int fw0 = u.f0 + 16 * warp;
        const int fl_last = u.f0 + u.nf - 1;                         // last valid flat row of the unit
        const int nvalid = max(0, min(16, u.f0 + u.nf - fw0));
        int prow[2]; bool rvalid[2]; int trow[2], hrow[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int f = fw0 + gid + 8 * r;
            rvalid[r] = f <= fl_last;
            int fc = min(f, fl_last);
            if (nvalid > 0) fc = min(f, fw0 + nvalid - 1);
            trow[r] = u.t_seq + fc / GQ; hrow[r] = u.hk * GQ + fc % GQ;
            prow[r] = u.pos_seq + fc / GQ;
        }
        const int pw_last = u.pos_seq + (fw0 + max(nvalid, 1) - 1) / GQ;   // largest position among the warp's rows
        const int pw_first = u.pos_seq + fw0 / GQ;
        const int p_unit_lo = u.pos_seq + u.f0 / GQ, p_unit_hi = u.pos_seq + fl_last / GQ;
        const int jlo = (KIND == 0) ? 0 : max(0, p_unit_lo - WINDOW + 1);
        const int jhi = p_unit_hi;
        const bool wact = nvalid > 0;

        // Q fragments (fp16), A layout, dims permuted within 8-groups for KIND 0 (slot 2t -> dim t, 2t+1 -> t+4)
        uint32_t qf[8][4];
        uint32_t rqf[4];
        {
            const __nv_bfloat16* q0 = P.q + ((long long)trow[0] * P.HL + hrow[0]) * DH;
            const __nv_bfloat16* q1 = P.q + ((long long)trow[1] * P.HL + hrow[1]) * DH;
            #pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                #pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    int d0, d1;
                    if (KIND == 0) { d0 = 16 * ks + 8 * hh + tig; d1 = d0 + 4; }
                    else { d0 = 16 * ks + 8 * hh + 2 * tig; d1 = d0 + 1; }
                    float a0 = rvalid[0] ? __bfloat162float(q0[d0]) : 0.f, a1 = rvalid[0] ? __bfloat162float(q0[d1]) : 0.f;
                    float b0 = rvalid[1] ? __bfloat162float(q1[d0]) : 0.f, b1 = rvalid[1] ? __bfloat162float(q1[d1]) : 0.f;
                    qf[ks][2 * hh] = pack_h2(a0, a1);
                    qf[ks][2 * hh + 1] = pack_h2(b0, b1);
                }
            }
            const uint32_t* r0 = reinterpret_cast<const uint32_t*>(P.relq + ((long long)trow[0] * P.HL + hrow[0]) * 16);
            const uint32_t* r1 = reinterpret_cast<const uint32_t*>(P.relq + ((long long)trow[1] * P.HL + hrow[1]) * 16);
            rqf[0] = rvalid[0] ? r0[tig] : 0u; rqf[1] = rvalid[1] ? r1[tig] : 0u;
            rqf[2] = rvalid[0] ? r0[tig + 4] : 0u; rqf[3] = rvalid[1] ? r1[tig + 4] : 0u;
        }
        const int dlt[2] = {pw_last - prow[0], pw_last - prow[1]};        // row shift in the bias staging (0..3)

        float o[16][4];
        #pragma unroll
        for (int i = 0; i < 16; i++) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
        float m_r[2], l_r[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (i0 == 0) {
                m_r[r] = __bfloat162float(P.sink[(long long)trow[r] * P.sink_stride + hrow[r]]) * 1.4426950408889634f;
                l_r[r] = (tig == 0) ? 1.f : 0.f;
            } else { m_r[r] = -1e30f; l_r[r] = 0.f; }
        }

        // pipeline prologue
        Raw<KIND> raw;
        load_raw<KIND>(raw, P, u, u.b_lo + i0, jlo, jhi);
        __syncthreads();                    // previous piece done with the buffers
        store_tile<KIND>(raw, Kbuf[0], Vbuf[0]);
        if (i0 + 1 < i1) load_raw<KIND>(raw, P, u, u.b_lo + i0 + 1, jlo, jhi);
        __syncthreads();

        for (int i = i0; i < i1; i++) {
            const int buf = (i - i0) & 1;
            if (i + 1 < i1) {
                store_tile<KIND>(raw, Kbuf[buf ^ 1], Vbuf[buf ^ 1]);
                if (i + 2 < i1) load_raw<KIND>(raw, P, u, u.b_lo + i + 2, jlo, jhi);
            }
            const int b = u.b_lo + i;
            const int j0 = b * BN;
            // warp-uniform block classification
            bool skip = !wact || (j0 > pw_last);
            if (KIND == 1) skip = skip || (j0 + BN - 1 < pw_first - WINDOW + 1);
            if (!skip) {
                const bool need_mask = (j0 + BN - 1 > pw_first) || (KIND == 1 && j0 < pw_last - WINDOW + 1);
                const bool need_bias = (pw_first - (j0 + BN - 1) < EXTENT);
                const uint32_t kb = smem_u32(Kbuf[buf]), vb = smem_u32(Vbuf[buf]);
                float s[8][4];
                #pragma unroll
                for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
                // S = Q K^T
                #pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    #pragma unroll
                    for (int np = 0; np < 4; np++) {
                        const int mi = lane >> 3, rr = lane & 7;
                        const int key = (2 * np + (mi >> 1)) * 8 + rr;
                        const int chunk = 2 * ks + (mi & 1);
                        uint32_t b0, b1, b2, b3;
                        ldsm_x4(b0, b1, b2, b3, kb + swz(key, chunk));
                        mma_f16(s[2 * np], qf[ks], b0, b1);
                        mma_f16(s[2 * np + 1], qf[ks], b2, b3);
                    }
                }
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) s[n][e] *= P.qk_scale;
                }
                if (need_bias) {
                    const int Dmax = pw_last - j0;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float wacc[5][4];
                        #pragma unroll
                        for (int k = 0; k < 5; k++) {
                            wacc[k][0] = wacc[k][1] = wacc[k][2] = wacc[k][3] = 0.f;
                            const int x = 32 * h + 8 * k + gid;
                            const int d = Dmax - x;
                            uint32_t b0 = 0u, b1 = 0u;
                            if (d >= 0 && d < EXTENT) {
                                const uint32_t* rp = reinterpret_cast<const uint32_t*>(P.relpT + d * 16);
                                b0 = __ldg(rp + tig); b1 = __ldg(rp + tig + 4);
                            }
                            mma_bf16(wacc[k], rqf, b0, b1);
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int k = 0; k < 5; k++) {
                            *reinterpret_cast<float2*>(Wsm + gid * WS + 8 * k + 2 * tig) =
                                make_float2(wacc[k][0] * 1.4426950408889634f, wacc[k][1] * 1.4426950408889634f);
                            *reinterpret_cast<float2*>(Wsm + (gid + 8) * WS + 8 * k + 2 * tig) =
                                make_float2(wacc[k][2] * 1.4426950408889634f, wacc[k][3] * 1.4426950408889634f);
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int nt = 0; nt < 4; nt++) {
                            const int n = 8 * nt + 2 * tig;
                            const float* w0 = Wsm + gid * WS + dlt[0] + n;
                            const float* w1 = Wsm + (gid + 8) * WS + dlt[1] + n;
                            s[4 * h + nt][0] += w0[0]; s[4 * h + nt][1] += w0[1];
                            s[4 * h + nt][2] += w1[0]; s[4 * h + nt][3] += w1[1];
                        }
                    }
                }
                if (need_mask) {
                    #pragma unroll
                    for (int n = 0; n < 8; n++) {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const int j = j0 + 8 * n + 2 * tig + (e & 1);
                            const int p = prow[e >> 1];
                            bool ok = j <= p;
                            if (KIND == 1) ok = ok && (p - j < WINDOW);
                            if (!ok) s[n][e] = -INFINITY;
                        }
                    }
                }
                // online softmax (rows gid, gid + 8)
                float alpha[2];
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    float mx = m_r[r];
                    #pragma unroll
                    for (int n = 0; n < 8; n++) mx = fmaxf(mx, fmaxf(s[n][2 * r], s[n][2 * r + 1]));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    alpha[r] = ex2(m_r[r] - mx);
                    m_r[r] = mx;
                    float ls = 0.f;
                    #pragma unroll
                    for (int n = 0; n < 8; n++) {
                        s[n][2 * r] = ex2(s[n][2 * r] - mx);
                        s[n][2 * r + 1] = ex2(s[n][2 * r + 1] - mx);
                        ls += s[n][2 * r] + s[n][2 * r + 1];
                    }
                    l_r[r] = l_r[r] * alpha[r] + ls;
                }
                if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        o[d][0] *= alpha[0]; o[d][1] *= alpha[0]; o[d][2] *= alpha[1]; o[d][3] *= alpha[1];
                    }
                }
                // O += P V
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    uint32_t a[4];
                    a[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
                    a[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
                    a[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                    a[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
                    #pragma unroll
                    for (int dp = 0; dp < 8; dp++) {
                        const int mi = lane >> 3, rr = lane & 7;
                        const int key = 16 * kk + (mi & 1) * 8 + rr;
                        const int chunk = 2 * dp + (mi >> 1);
                        uint32_t b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3, vb + swz(key, chunk));
                        mma_f16(o[2 * dp], a, b0, b1);
                        mma_f16(o[2 * dp + 1], a, b2, b3);
                    }
                }
            }
            __syncthreads();
        }

        // epilogue: row sums, normalize, rotate back (global), gate (whole unit) or partial
        float inv[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            float l = l_r[r];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            l_r[r] = l;
            inv[r] = l > 0.f ? 1.f / l : 0.f;
        }
        if (!wact) continue;
        float res[16][4];
        if (KIND == 0) {
            uint32_t af[8][4];
            #pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                af[ks][0] = pack_h2(o[2 * ks][0] * inv[0], o[2 * ks][1] * inv[0]);
                af[ks][1] = pack_h2(o[2 * ks][2] * inv[1], o[2 * ks][3] * inv[1]);
                af[ks][2] = pack_h2(o[2 * ks + 1][0] * inv[0], o[2 * ks + 1][1] * inv[0]);
                af[ks][3] = pack_h2(o[2 * ks + 1][2] * inv[1], o[2 * ks + 1][3] * inv[1]);
            }
            #pragma unroll
            for (int jt = 0; jt < 16; jt++) {
                res[jt][0] = res[jt][1] = res[jt][2] = res[jt][3] = 0.f;
                const uint32_t* mrow = reinterpret_cast<const uint32_t*>(P.rot + (8 * jt + gid) * DH);
                #pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    uint32_t b0 = __ldg(mrow + 8 * ks + tig), b1 = __ldg(mrow + 8 * ks + 4 + tig);
                    mma_f16(res[jt], af[ks], b0, b1);
                }
                #pragma unroll
                for (int e = 0; e < 4; e++) res[jt][e] *= 0.08838834764831845f;
            }
        } else {
            #pragma unroll
            for (int d = 0; d < 16; d++) {
                res[d][0] = o[d][0] * inv[0]; res[d][1] = o[d][1] * inv[0];
                res[d][2] = o[d][2] * inv[1]; res[d][3] = o[d][3] * inv[1];
            }
        }
        if (whole) {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (!rvalid[r]) continue;
                const __nv_bfloat16* g = P.gate + (long long)trow[r] * P.gate_stride + hrow[r] * DH;
                __nv_bfloat16* op = P.out + ((long long)trow[r] * P.HL + hrow[r]) * DH;
                #pragma unroll
                for (int d = 0; d < 16; d++) {
                    const int dim = 8 * d + 2 * tig;
                    __nv_bfloat162 gg = *reinterpret_cast<const __nv_bfloat162*>(g + dim);
                    float v0 = __bfloat162float(__float2bfloat16_rn(res[d][2 * r]));
                    float v1 = __bfloat162float(__float2bfloat16_rn(res[d][2 * r + 1]));
                    v0 *= 1.f / (1.f + __expf(-__bfloat162float(gg.x)));
                    v1 *= 1.f / (1.f + __expf(-__bfloat162float(gg.y)));
                    *reinterpret_cast<__nv_bfloat162*>(op + dim) = __floats2bfloat162_rn(v0, v1);
                }
            }
        } else {
            const int slot = (u_idx == u_first) ? 0 : 1;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                const int ir = 16 * warp + gid + 8 * r;
                float* po = P.part_o + (((long long)c * 2 + slot) * BM + ir) * DH;
                #pragma unroll
                for (int d = 0; d < 16; d++)
                    *reinterpret_cast<float2*>(po + 8 * d + 2 * tig) = make_float2(res[d][2 * r], res[d][2 * r + 1]);
                if (tig == 0) {
                    float* pm = P.part_ml + (((long long)c * 2 + slot) * BM + ir) * 2;
                    pm[0] = m_r[r]; pm[1] = l_r[r];
                }
            }
        }
    }
}

// merge the partials of units cut by segment boundaries; one CTA per unit, 256 threads (row = tid / 2, 64 dims each)
__global__ void __launch_bounds__(256) pattn_merge(const Params P) {
    const int u_idx = blockIdx.x;
    const Unit u = P.units[u_idx];
    const int ucum = P.cum[u_idx], unext = P.cum[u_idx + 1];
    const int c_lo = ucum / P.seg, c_hi = (unext - 1) / P.seg;
    if (c_lo == c_hi) return;
    const int ir = threadIdx.x >> 1, half = threadIdx.x & 1;
    if (ir >= u.nf) return;
    float M = -1e30f;
    for (int c = c_lo; c <= c_hi; c++) {
        const int slot = (c > c_lo || ucum <= c * P.seg) ? 0 : 1;
        M = fmaxf(M, P.part_ml[(((long long)c * 2 + slot) * BM + ir) * 2]);
    }
    float acc[64];
    #pragma unroll
    for (int d = 0; d < 64; d++) acc[d] = 0.f;
    float den = 0.f;
    for (int c = c_lo; c <= c_hi; c++) {
        const int slot = (c > c_lo || ucum <= c * P.seg) ? 0 : 1;
        const float* pm = P.part_ml + (((long long)c * 2 + slot) * BM + ir) * 2;
        const float wgt = pm[1] * exp2f(pm[0] - M);
        den += wgt;
        const float4* po = reinterpret_cast<const float4*>(P.part_o + (((long long)c * 2 + slot) * BM + ir) * DH + half * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            float4 v = po[d];
            acc[4 * d] += wgt * v.x; acc[4 * d + 1] += wgt * v.y; acc[4 * d + 2] += wgt * v.z; acc[4 * d + 3] += wgt * v.w;
        }
    }
    const float inv = den > 0.f ? 1.f / den : 0.f;
    const int f = u.f0 + ir;
    const int t = u.t_seq + f / GQ, h = u.hk * GQ + f % GQ;
    const __nv_bfloat16* g = P.gate + (long long)t * P.gate_stride + h * DH + half * 64;
    __nv_bfloat16* op = P.out + ((long long)t * P.HL + h) * DH + half * 64;
    #pragma unroll
    for (int d = 0; d < 64; d += 2) {
        float v0 = __bfloat162float(__float2bfloat16_rn(acc[d] * inv));
        float v1 = __bfloat162float(__float2bfloat16_rn(acc[d + 1] * inv));
        v0 *= 1.f / (1.f + __expf(-__bfloat162float(g[d])));
        v1 *= 1.f / (1.f + __expf(-__bfloat162float(g[d + 1])));
        *reinterpret_cast<__nv_bfloat162*>(op + d) = __floats2bfloat162_rn(v0, v1);
    }
}

static const int SMEM_BYTES = 4 * 16384 + NWARP * 16 * WS * 4;

void pattn(int64_t kind, torch::Tensor q, torch::Tensor relq, torch::Tensor relpT, torch::Tensor sink, torch::Tensor gate,
           torch::Tensor kq, torch::Tensor vq, torch::Tensor ks, torch::Tensor vs, torch::Tensor page_table, int64_t ring,
           torch::Tensor rot, torch::Tensor units, torch::Tensor cum, int64_t n_units, int64_t seg, int64_t total, int64_t nCTA,
           torch::Tensor out, torch::Tensor part_o, torch::Tensor part_ml) {
    Params P;
    P.q = (const __nv_bfloat16*)q.data_ptr(); P.relq = (const __nv_bfloat16*)relq.data_ptr();
    P.relpT = (const __nv_bfloat16*)relpT.data_ptr();
    P.sink = (const __nv_bfloat16*)sink.data_ptr(); P.sink_stride = sink.stride(0);
    P.gate = (const __nv_bfloat16*)gate.data_ptr(); P.gate_stride = gate.stride(0);
    P.kq = (const uint8_t*)kq.data_ptr(); P.vq = (const uint8_t*)vq.data_ptr();
    P.ks = (const __half*)ks.data_ptr(); P.vs = (const __half*)vs.data_ptr();
    P.cs_pos = kq.stride(-3); P.cs_h = kq.stride(-2);
    P.sc_pos = kind == 0 ? ks.stride(-3) : ks.stride(-2);
    P.page_table = (const int*)page_table.data_ptr(); P.pt_stride = page_table.stride(0); P.ring = ring;
    P.rot = (const __half*)rot.data_ptr();
    P.units = (const Unit*)units.data_ptr(); P.cum = (const int*)cum.data_ptr(); P.n_units = n_units; P.seg = seg; P.total = total;
    P.out = (__nv_bfloat16*)out.data_ptr(); P.part_o = (float*)part_o.data_ptr(); P.part_ml = (float*)part_ml.data_ptr();
    P.HL = q.size(1);
    P.qk_scale = 0.08838834764831845f * 1.4426950408889634f;
    auto stream = c10::cuda::getCurrentCUDAStream();
    static bool init[2] = {false, false};
    if (kind == 0) {
        if (!init[0]) { cudaFuncSetAttribute(pattn_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); init[0] = true; }
        pattn_kernel<0><<<nCTA, NTHR, SMEM_BYTES, stream>>>(P);
    } else {
        if (!init[1]) { cudaFuncSetAttribute(pattn_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); init[1] = true; }
        pattn_kernel<1><<<nCTA, NTHR, SMEM_BYTES, stream>>>(P);
    }
    if (nCTA > 1) pattn_merge<<<n_units, 256, 0, stream>>>(P);
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void pattn(int64_t kind, torch::Tensor q, torch::Tensor relq, torch::Tensor relpT, torch::Tensor sink, torch::Tensor gate,
           torch::Tensor kq, torch::Tensor vq, torch::Tensor ks, torch::Tensor vs, torch::Tensor page_table, int64_t ring,
           torch::Tensor rot, torch::Tensor units, torch::Tensor cum, int64_t n_units, int64_t seg, int64_t total, int64_t nCTA,
           torch::Tensor out, torch::Tensor part_o, torch::Tensor part_ml);
"""
