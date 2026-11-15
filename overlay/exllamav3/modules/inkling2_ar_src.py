"""CUDA source of the Inkling-2 PCIe peer all-reduce (inkling2_ar.py JIT-builds it). Kept as Python so every install
(including pip installs that drop non-Python files) ships it."""
SRC = r"""// Inkling-2: graph-safe PCIe peer-to-peer all-reduce (recursive doubling, push design) for the small per-layer messages
// of the speculative decode step (fp16/bf16/fp32 [T, 4608], T <= ~100), W = 2 or 4. In round k every rank pushes its
// (partial) message into its partner's (rank ^ (1 << k)) receive slot with posted P2P writes, raises the partner's
// per-block flag, waits for its own flag, and adds the received partial in fp32. Epochs live in a per-block device
// counter, so the kernel works unchanged inside CUDA graphs. Two parity buffers keep a fast rank from overwriting a slot
// its partner still reads: call e+2 on rank A needs the partner's flag of call e+1, raised only after its call e ended.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int MAXW = 8;
constexpr int MAXB = 128;

struct Peers { char* buf[MAXW]; unsigned* flag[MAXW]; };

struct State {
    int rank, world, device;
    size_t cap;
    char* buf; unsigned* flag; unsigned* counter; unsigned* gbar;
    Peers peers;
    std::vector<void*> opened;
};
static std::vector<State> g_states;

__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v)
{
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p)
{
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <typename T> struct Vec;
template <> struct Vec<half> {
    static constexpr int N = 8;
    __device__ static void add(float* a, uint4 v) { const half2* h = (const half2*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float2 f = __half22float2(h[i]); a[2 * i] += f.x; a[2 * i + 1] += f.y; } }
    __device__ static uint4 pack(const float* a) { uint4 v; half2* h = (half2*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(a[2 * i], a[2 * i + 1]); return v; }
};
template <> struct Vec<__nv_bfloat16> {
    static constexpr int N = 8;
    __device__ static void add(float* a, uint4 v) { const __nv_bfloat162* h = (const __nv_bfloat162*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(h[i]); a[2 * i] += f.x; a[2 * i + 1] += f.y; } }
    __device__ static uint4 pack(const float* a) { uint4 v; __nv_bfloat162* h = (__nv_bfloat162*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]); return v; }
};
template <> struct Vec<float> {
    static constexpr int N = 4;
    __device__ static void add(float* a, uint4 v) { const float* f = (const float*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] += f[i]; }
    __device__ static uint4 pack(const float* a) { uint4 v; float* f = (float*) &v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) f[i] = a[i]; return v; }
};

// Recursive doubling, W = 2 or 4: round k exchanges the (partial) message with partner rank ^ (1 << k). Each GPU has one
// outgoing and one incoming P2P flow per round; with all-to-all pushes (3 flows per GPU) this root complex drops to
// ~3 GB/s per GPU, with one flow each way it keeps ~45 GB/s. Partials are added in fp32 (a + b == b + a, so all ranks
// end bitwise identical). Round-2 payload: fp32 partials (F32MID, no intermediate rounding) or the input dtype (the
// partial rounded once on both sides of the pair, as NCCL's per-hop rounding).
__device__ __forceinline__ void grid_barrier(unsigned* gbar, int r, unsigned epoch)
{
    // all blocks of this GPU leave round r together, so its P2P writes of the next round go to one peer at a time.
    // Generation barrier (count, gen) at gbar[2 r], gbar[2 r + 1]: independent of grid size and of earlier calls
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* cnt = gbar + 2 * r;
        unsigned* gen = gbar + 2 * r + 1;
        const unsigned g0 = ld_acquire_sys(gen);
        __threadfence();
        if (atomicAdd(cnt, 1u) == gridDim.x - 1) {
            atomicExch(cnt, 0u);
            __threadfence();
            st_release_sys(gen, g0 + 1);
        } else {
            while (ld_acquire_sys(gen) == g0) {}
        }
    }
    __syncthreads();
}

constexpr int VPT = 4;       // 16-byte vectors per thread per call

template <typename T, int W, bool F32MID>
__global__ void __launch_bounds__(256) ar_rd_kernel(uint4* __restrict__ x, int n_vec, Peers peers, unsigned* counter, int rank, size_t cap, unsigned* gbar, int use_gb)
{
    constexpr int N = Vec<T>::N;                 // elements per 16-byte vector
    constexpr int ROUNDS = W == 2 ? 1 : 2;
    const int b = blockIdx.x, nb = gridDim.x;
    __shared__ unsigned s_epoch;
    if (threadIdx.x == 0) s_epoch = counter[b] + 1;
    __syncthreads();
    const unsigned epoch = s_epoch;
    const int par = epoch & 1;
    const int per = (n_vec + nb - 1) / nb;
    const int v0 = b * per, v1 = min(n_vec, v0 + per);
    float acc[VPT][N];
    #pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int v = v0 + j * blockDim.x + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < N; ++i) acc[j][i] = 0.0f;
        if (v < v1) Vec<T>::add(acc[j], x[v]);
    }
    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int partner = rank ^ (1 << r);
        const bool f32 = F32MID && r > 0;          // payload of this round in fp32
        // slot layout per rank: [parity][round][cap]
        char* dst = peers.buf[partner] + (size_t) (par * 2 + r) * cap;
        const char* src = peers.buf[rank] + (size_t) (par * 2 + r) * cap;
        #pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int v = v0 + j * blockDim.x + threadIdx.x;
            if (v >= v1) continue;
            if (f32) {
                #pragma unroll
                for (int q = 0; q < N / 4; ++q)
                    ((uint4*) dst)[q * n_vec + v] = Vec<float>::pack(acc[j] + 4 * q);     // planar: contiguous warp stores
            } else {
                uint4 d = Vec<T>::pack(acc[j]);
                ((uint4*) dst)[v] = d;
                if (r > 0) {                       // use the rounded partial on both sides of the pair
                    #pragma unroll
                    for (int i = 0; i < N; ++i) acc[j][i] = 0.0f;
                    Vec<T>::add(acc[j], d);
                }
            }
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            st_release_sys(peers.flag[partner] + (par * 2 + r) * MAXB + b, epoch);
            const unsigned* f = peers.flag[rank] + (par * 2 + r) * MAXB + b;
            while (ld_acquire_sys(f) != epoch) {}
        }
        __syncthreads();
        if (use_gb && r + 1 < ROUNDS) grid_barrier(gbar, 4 + r, epoch);
        #pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int v = v0 + j * blockDim.x + threadIdx.x;
            if (v >= v1) continue;
            if (f32) {
                #pragma unroll
                for (int q = 0; q < N / 4; ++q)
                    Vec<float>::add(acc[j] + 4 * q, __ldcg(((const uint4*) src) + q * n_vec + v));
            } else
                Vec<T>::add(acc[j], __ldcg(((const uint4*) src) + v));
        }
    }
    #pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int v = v0 + j * blockDim.x + threadIdx.x;
        if (v < v1) x[v] = Vec<T>::pack(acc[j]);
    }
    if (threadIdx.x == 0) counter[b] = epoch;
}

// Large messages (prefill, [512, 4608] = 4.7 MB): reduce-scatter + all-gather by recursive halving / doubling, W = 4,
// one P2P flow per GPU per round, 1.5x the message moved per GPU. Quarters of the message: rank r owns quarter
// 2h + b (h = r & 1, b = (r >> 1) & 1). R1 (partner r^1) halves, R2 (r^2) quarters -> the owned quarter is final
// (computed by exactly one rank, so every rank ends bitwise identical), R3 (r^2) and R4 (r^1) gather it back.
// Slots per rank: [parity][round 0..3][cap], cap >= half the message. Each block owns the same slice of every quarter.
constexpr int VPT2 = 4;

__device__ __forceinline__ void round_sync(Peers& peers, int partner, int rank, int par, int r, int b, unsigned epoch)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        st_release_sys(peers.flag[partner] + (par * 4 + r) * MAXB + b, epoch);
        const unsigned* f = peers.flag[rank] + (par * 4 + r) * MAXB + b;
        while (ld_acquire_sys(f) != epoch) {}
    }
    __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(256) ar_rsag_kernel(uint4* __restrict__ x, int n_vec, Peers peers, unsigned* counter, int rank, size_t cap, unsigned* gbar, int use_gb)
{
    constexpr int N = Vec<T>::N;
    const int b = blockIdx.x, nb = gridDim.x;
    __shared__ unsigned s_epoch;
    if (threadIdx.x == 0) s_epoch = counter[b] + 1;
    __syncthreads();
    const unsigned epoch = s_epoch;
    const int par = epoch & 1;
    const int Q = n_vec / 4;
    const int per = (Q + nb - 1) / nb;
    const int k0 = b * per, k1 = min(Q, k0 + per);
    const int h = rank & 1, bq = (rank >> 1) & 1;
    const int p1 = rank ^ 1, p2 = rank ^ 2;
    auto slot = [&](int who, int r) { return (uint4*) (peers.buf[who] + (size_t) (par * 4 + r) * cap); };
    // R1: send the other half (2 quarters) to p1; receive p1's copy of my half
    #pragma unroll
    for (int j = 0; j < VPT2; ++j) {
        const int k = k0 + j * blockDim.x + threadIdx.x;
        if (k >= k1) continue;
        slot(p1, 0)[k] = x[(2 * (1 - h)) * Q + k];
        slot(p1, 0)[Q + k] = x[(2 * (1 - h) + 1) * Q + k];
    }
    round_sync(peers, p1, rank, par, 0, b, epoch);
    if (use_gb) grid_barrier(gbar, 0, epoch);
    float own[VPT2][N];
    #pragma unroll
    for (int j = 0; j < VPT2; ++j) {
        const int k = k0 + j * blockDim.x + threadIdx.x;
        if (k >= k1) continue;
        // partial of the quarter p2 finishes: sent rounded to T
        float o[N];
        #pragma unroll
        for (int i = 0; i < N; ++i) { o[i] = 0.0f; own[j][i] = 0.0f; }
        Vec<T>::add(o, x[(2 * h + (1 - bq)) * Q + k]);
        Vec<T>::add(o, __ldcg(slot(rank, 0) + (1 - bq) * Q + k));
        slot(p2, 1)[k] = Vec<T>::pack(o);
        Vec<T>::add(own[j], x[(2 * h + bq) * Q + k]);
        Vec<T>::add(own[j], __ldcg(slot(rank, 0) + bq * Q + k));
    }
    round_sync(peers, p2, rank, par, 1, b, epoch);
    if (use_gb) grid_barrier(gbar, 1, epoch);
    // final owned quarter -> x and to p2
    #pragma unroll
    for (int j = 0; j < VPT2; ++j) {
        const int k = k0 + j * blockDim.x + threadIdx.x;
        if (k >= k1) continue;
        Vec<T>::add(own[j], __ldcg(slot(rank, 1) + k));
        uint4 f = Vec<T>::pack(own[j]);
        x[(2 * h + bq) * Q + k] = f;
        slot(p2, 2)[k] = f;
    }
    round_sync(peers, p2, rank, par, 2, b, epoch);
    if (use_gb) grid_barrier(gbar, 2, epoch);
    // R4: my half (owned quarter + p2's quarter) to p1; take the other half from p1
    #pragma unroll
    for (int j = 0; j < VPT2; ++j) {
        const int k = k0 + j * blockDim.x + threadIdx.x;
        if (k >= k1) continue;
        uint4 g = __ldcg(slot(rank, 2) + k);
        x[(2 * h + (1 - bq)) * Q + k] = g;
        slot(p1, 3)[bq * Q + k] = x[(2 * h + bq) * Q + k];
        slot(p1, 3)[(1 - bq) * Q + k] = g;
    }
    round_sync(peers, p1, rank, par, 3, b, epoch);
    #pragma unroll
    for (int j = 0; j < VPT2; ++j) {
        const int k = k0 + j * blockDim.x + threadIdx.x;
        if (k >= k1) continue;
        x[(2 * (1 - h)) * Q + k] = __ldcg(slot(rank, 3) + k);
        x[(2 * (1 - h) + 1) * Q + k] = __ldcg(slot(rank, 3) + Q + k);
    }
    if (threadIdx.x == 0) counter[b] = epoch;
}

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, "inkling2_ar: ", cudaGetErrorString(e_)); } while (0)

int64_t ar_init(int64_t rank, int64_t world, int64_t cap_bytes)
{
    TORCH_CHECK(world == 2 || world == 4, "inkling2_ar: world 2 or 4");
    TORCH_CHECK(cap_bytes % 16 == 0);
    State s{};
    s.rank = rank; s.world = world; s.cap = cap_bytes;
    CUDA_CHECK(cudaGetDevice(&s.device));
    CUDA_CHECK(cudaMalloc(&s.buf, 8 * cap_bytes));
    CUDA_CHECK(cudaMalloc(&s.flag, 8 * MAXB * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&s.counter, MAXB * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&s.gbar, 16 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(s.gbar, 0, 16 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(s.buf, 0, 8 * cap_bytes));
    CUDA_CHECK(cudaMemset(s.flag, 0, 8 * MAXB * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(s.counter, 0, MAXB * sizeof(unsigned)));
    CUDA_CHECK(cudaDeviceSynchronize());
    g_states.push_back(s);
    return (int64_t) g_states.size() - 1;
}

torch::Tensor ar_handles(int64_t h)
{
    State& s = g_states[h];
    cudaIpcMemHandle_t hb, hf;
    CUDA_CHECK(cudaIpcGetMemHandle(&hb, s.buf));
    CUDA_CHECK(cudaIpcGetMemHandle(&hf, s.flag));
    auto t = torch::empty({2 * (int64_t) sizeof(cudaIpcMemHandle_t)}, torch::kUInt8);
    memcpy(t.data_ptr(), &hb, sizeof(hb));
    memcpy((char*) t.data_ptr() + sizeof(hb), &hf, sizeof(hf));
    return t;
}

void ar_open(int64_t h, torch::Tensor all)       // all: [world, 2 * 64] uint8 (cpu)
{
    State& s = g_states[h];
    TORCH_CHECK(all.size(0) == s.world);
    for (int p = 0; p < s.world; ++p) {
        if (p == s.rank) { s.peers.buf[p] = s.buf; s.peers.flag[p] = s.flag; continue; }
        cudaIpcMemHandle_t hb, hf;
        const char* src = (const char*) all[p].contiguous().data_ptr();
        memcpy(&hb, src, sizeof(hb)); memcpy(&hf, src + sizeof(hb), sizeof(hf));
        void *pb, *pf;
        CUDA_CHECK(cudaIpcOpenMemHandle(&pb, hb, cudaIpcMemLazyEnablePeerAccess));
        CUDA_CHECK(cudaIpcOpenMemHandle(&pf, hf, cudaIpcMemLazyEnablePeerAccess));
        s.peers.buf[p] = (char*) pb; s.peers.flag[p] = (unsigned*) pf;
        s.opened.push_back(pb); s.opened.push_back(pf);
    }
}

void ar_all_reduce(int64_t h, torch::Tensor x, int64_t blocks, int64_t threads, int64_t f32mid, int64_t use_gb)
{
    State& s = g_states[h];
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    const size_t bytes = x.numel() * x.element_size();
    const bool f32 = f32mid && x.scalar_type() != at::kFloat;
    TORCH_CHECK(bytes % 16 == 0 && bytes * (f32 ? 2 : 1) <= s.cap, "inkling2_ar: message ", bytes, " B (cap ", s.cap, ")");
    const int n_vec = bytes / 16;
    const int th = (int) threads;
    int nb = (int) blocks;
    const int nmin = (n_vec + th * VPT - 1) / (th * VPT);
    if (nb <= 0) nb = nmin;
    nb = std::max(nb, nmin);
    nb = std::min(nb, std::max(1, n_vec));
    // per-GPU barrier between the 2 rounds only from 256 KB (costs ~1 us; T=48 fp16: 27.9 vs 34 us, T=5: 8.0 vs 7.0 us)
    if (bytes < (256 << 10)) use_gb = 0;
    TORCH_CHECK(nb <= MAXB, "inkling2_ar: message too large for ", MAXB, " blocks");
    auto stream = c10::cuda::getCurrentCUDAStream();
    uint4* p = (uint4*) x.data_ptr();
#define LAUNCH(T, W, F) ar_rd_kernel<T, W, F><<<nb, th, 0, stream>>>(p, n_vec, s.peers, s.counter, s.rank, s.cap, s.gbar, (int) use_gb)
#define BY_W(T, F) if (s.world == 2) LAUNCH(T, 2, F); else LAUNCH(T, 4, F)
    if (x.scalar_type() == at::kHalf) { if (f32) { BY_W(half, true); } else { BY_W(half, false); } }
    else if (x.scalar_type() == at::kBFloat16) { if (f32) { BY_W(__nv_bfloat16, true); } else { BY_W(__nv_bfloat16, false); } }
    else if (x.scalar_type() == at::kFloat) { BY_W(float, false); }
    else TORCH_CHECK(false, "inkling2_ar: dtype");
    CUDA_CHECK(cudaGetLastError());
}

void ar_all_reduce_big(int64_t h, torch::Tensor x, int64_t threads, int64_t use_gb)
{
    State& s = g_states[h];
    TORCH_CHECK(s.world == 4, "inkling2_ar: two-shot path is W = 4");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    const size_t bytes = x.numel() * x.element_size();
    TORCH_CHECK(bytes % 64 == 0 && bytes / 2 <= s.cap, "inkling2_ar: big message ", bytes, " B (slot ", s.cap, ")");
    const int n_vec = bytes / 16, Q = n_vec / 4, th = (int) threads;
    const int nb = MAXB;            // constant grid: the per-round grid barrier counts epoch * gridDim.x arrivals
    TORCH_CHECK((int64_t) nb * th * VPT2 >= Q, "inkling2_ar: message too large");
    auto stream = c10::cuda::getCurrentCUDAStream();
    uint4* p = (uint4*) x.data_ptr();
    if (x.scalar_type() == at::kHalf) ar_rsag_kernel<half><<<nb, th, 0, stream>>>(p, n_vec, s.peers, s.counter, s.rank, s.cap, s.gbar, (int) use_gb);
    else if (x.scalar_type() == at::kBFloat16) ar_rsag_kernel<__nv_bfloat16><<<nb, th, 0, stream>>>(p, n_vec, s.peers, s.counter, s.rank, s.cap, s.gbar, (int) use_gb);
    else if (x.scalar_type() == at::kFloat) ar_rsag_kernel<float><<<nb, th, 0, stream>>>(p, n_vec, s.peers, s.counter, s.rank, s.cap, s.gbar, (int) use_gb);
    else TORCH_CHECK(false, "inkling2_ar: dtype");
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &ar_init);
    m.def("handles", &ar_handles);
    m.def("open", &ar_open);
    m.def("all_reduce", &ar_all_reduce);
    m.def("all_reduce_big", &ar_all_reduce_big);
}
"""
