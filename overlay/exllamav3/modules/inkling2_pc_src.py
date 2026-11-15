"""CUDA source of the Inkling-2 prefill copy-engine collectives (inkling2_pc.py JIT-builds it; kept as Python so pip
installs ship it). Data moves with cudaMemcpyAsync peer copies (copy engines, no SMs) into IPC-mapped slabs; cross-GPU
ordering uses 32-bit flags written with cuStreamWriteValue32 into the receiver's flag array right after the copy on
the same stream, and waited on with cuStreamWaitValue32 (GEQ) by the receiver's stream: both are front-end stream
memory operations, so a collective holds no SM while it waits or moves bytes."""
SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, "inkling2_pc: ", #x, ": ", cudaGetErrorString(e_)); } while (0)
#define DK(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(e_, &s_); TORCH_CHECK(false, "inkling2_pc: ", #x, ": ", s_ ? s_ : "?"); } } while (0)

constexpr int MAXW = 8;
struct St {
    int rank, world, dev; size_t bytes; int nflag;
    char* slab; unsigned* flag;
    char* pslab[MAXW]; unsigned* pflag[MAXW];
    int flush;
};
static std::vector<St> g;

int64_t pc_init(int64_t rank, int64_t world, int64_t bytes, int64_t nflag)
{
    St s{}; s.rank = rank; s.world = world; s.bytes = bytes; s.nflag = nflag;
    CK(cudaGetDevice(&s.dev));
    CK(cudaMalloc(&s.slab, bytes));
    CK(cudaMalloc(&s.flag, nflag * sizeof(unsigned)));
    CK(cudaMemset(s.flag, 0, nflag * sizeof(unsigned)));
    CK(cudaMemset(s.slab, 0, bytes));
    CK(cudaDeviceSynchronize());
    CUdevice d; DK(cuDeviceGet(&d, s.dev));
    int fl = 0; cuDeviceGetAttribute(&fl, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, d);
    s.flush = fl;
    g.push_back(s);
    return (int64_t) g.size() - 1;
}

torch::Tensor pc_handles(int64_t h)
{
    St& s = g[h];
    cudaIpcMemHandle_t a, b;
    CK(cudaIpcGetMemHandle(&a, s.slab)); CK(cudaIpcGetMemHandle(&b, s.flag));
    auto t = torch::empty({2 * (int64_t) sizeof(cudaIpcMemHandle_t)}, torch::kUInt8);
    memcpy(t.data_ptr(), &a, sizeof(a)); memcpy((char*) t.data_ptr() + sizeof(a), &b, sizeof(b));
    return t;
}

void pc_open(int64_t h, torch::Tensor all)
{
    St& s = g[h];
    for (int p = 0; p < s.world; ++p) {
        if (p == s.rank) { s.pslab[p] = s.slab; s.pflag[p] = s.flag; continue; }
        cudaIpcMemHandle_t a, b;
        const char* src = (const char*) all[p].contiguous().data_ptr();
        memcpy(&a, src, sizeof(a)); memcpy(&b, src + sizeof(a), sizeof(b));
        void *pa, *pb;
        CK(cudaIpcOpenMemHandle(&pa, a, cudaIpcMemLazyEnablePeerAccess));
        CK(cudaIpcOpenMemHandle(&pb, b, cudaIpcMemLazyEnablePeerAccess));
        s.pslab[p] = (char*) pa; s.pflag[p] = (unsigned*) pb;
    }
}

int64_t pc_base(int64_t h, int64_t rank) { return (int64_t) g[h].pslab[rank]; }
int64_t pc_flush(int64_t h) { return g[h].flush; }

// local slab view as a tensor (no ownership)
torch::Tensor pc_view(int64_t h, int64_t off, std::vector<int64_t> shape, int64_t dtype_code)
{
    St& s = g[h];
    auto dt = dtype_code == 0 ? torch::kBFloat16 : dtype_code == 1 ? torch::kHalf : dtype_code == 2 ? torch::kFloat : dtype_code == 3 ? torch::kInt : torch::kUInt8;
    int64_t n = 1; for (auto x : shape) n *= x;
    TORCH_CHECK(off >= 0 && off + n * (int64_t) torch::elementSize(dt) <= (int64_t) s.bytes, "inkling2_pc: view out of slab");
    return torch::from_blob(s.slab + off, shape, torch::TensorOptions().dtype(dt).device(torch::kCUDA, s.dev));
}

// copy nbytes from a local device pointer to (dst rank's slab + dst_off), 2D when height > 1 (pitches in bytes)
void pc_copy(int64_t h, int64_t dst, int64_t dst_off, int64_t src, int64_t width, int64_t height, int64_t dpitch, int64_t spitch, int64_t stream)
{
    St& s = g[h];
    cudaStream_t st = (cudaStream_t) stream;
    char* d = s.pslab[dst] + dst_off;
    if (height <= 1) CK(cudaMemcpyAsync(d, (const void*) src, width, cudaMemcpyDeviceToDevice, st));
    else CK(cudaMemcpy2DAsync(d, dpitch, (const void*) src, spitch, width, height, cudaMemcpyDeviceToDevice, st));
}

void pc_signal(int64_t h, int64_t dst, int64_t idx, int64_t value, int64_t stream)
{
    St& s = g[h];
    DK(cuStreamWriteValue32((CUstream) stream, (CUdeviceptr) (s.pflag[dst] + idx), (cuuint32_t) value, 0));
}

void pc_wait(int64_t h, int64_t idx, int64_t value, int64_t stream)
{
    St& s = g[h];
    unsigned f = CU_STREAM_WAIT_VALUE_GEQ | (s.flush ? CU_STREAM_WAIT_VALUE_FLUSH : 0);
    DK(cuStreamWaitValue32((CUstream) stream, (CUdeviceptr) (s.flag + idx), (cuuint32_t) value, f));
}

// out[i] = sum_k in_k[i] (fp32 accumulation, k = 1..4, inputs fp16 or bf16 or fp32 by code, out by code)
template <typename T> __device__ __forceinline__ float ld(const T* p, int64_t i);
template <> __device__ __forceinline__ float ld<half>(const half* p, int64_t i) { return __half2float(p[i]); }
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
template <> __device__ __forceinline__ float ld<float>(const float* p, int64_t i) { return p[i]; }
template <typename T> __device__ __forceinline__ T cv(float x);
template <> __device__ __forceinline__ half cv<half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cv<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ float cv<float>(float x) { return x; }

template <typename TI, typename TO>
__global__ void sum_kernel(TO* __restrict__ out, const TI* __restrict__ a, const TI* __restrict__ b, const TI* __restrict__ c,
                           const TI* __restrict__ d, int n_in, int64_t n)
{
    // 8 elements per thread-step, 16-byte aligned (n % 8 == 0 enforced on the host)
    for (int64_t v = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; v < n / 8; v += (int64_t) gridDim.x * blockDim.x) {
        float acc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = ld(a, v * 8 + j) + ld(b, v * 8 + j);
        if (n_in > 2) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += ld(c, v * 8 + j);
        }
        if (n_in > 3) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += ld(d, v * 8 + j);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) out[v * 8 + j] = cv<TO>(acc[j]);
    }
}

void pc_sum(torch::Tensor out, std::vector<torch::Tensor> ins)
{
    TORCH_CHECK(ins.size() >= 2 && ins.size() <= 4);
    const int64_t n = out.numel();
    TORCH_CHECK(n % 8 == 0 && out.is_contiguous());
    for (auto& t : ins) TORCH_CHECK(t.numel() == n && t.is_contiguous() && t.scalar_type() == ins[0].scalar_type());
    auto st = c10::cuda::getCurrentCUDAStream();
    int th = 256; int64_t nb = std::min<int64_t>((n / 8 + th - 1) / th, 188 * 8);
    const int ni = ins.size();
    auto P = [&](int k) { return ins[std::min<int>(k, ni - 1)].data_ptr(); };
#define SUMK(TI, TO) sum_kernel<TI, TO><<<nb, th, 0, st>>>((TO*) out.data_ptr(), (const TI*) P(0), (const TI*) P(1), (const TI*) P(2), (const TI*) P(3), ni, n)
    auto ti = ins[0].scalar_type(), to = out.scalar_type();
    if (ti == at::kHalf && to == at::kHalf) SUMK(half, half);
    else if (ti == at::kHalf && to == at::kFloat) SUMK(half, float);
    else if (ti == at::kBFloat16 && to == at::kBFloat16) SUMK(__nv_bfloat16, __nv_bfloat16);
    else if (ti == at::kFloat && to == at::kFloat) SUMK(float, float);
    else if (ti == at::kFloat && to == at::kHalf) SUMK(float, half);
    else TORCH_CHECK(false, "inkling2_pc: sum dtypes");
    CK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &pc_init); m.def("handles", &pc_handles); m.def("open", &pc_open); m.def("base", &pc_base);
    m.def("flush", &pc_flush); m.def("view", &pc_view); m.def("copy", &pc_copy); m.def("signal", &pc_signal);
    m.def("wait", &pc_wait); m.def("sum", &pc_sum);
}
"""
