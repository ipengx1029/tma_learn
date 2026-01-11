#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "vector"
#include "iomanip"
#include "algorithm"
#include <random>

#define checkCudaError(call) \
    do {    \
        cudaError_t e = call;   \
        if (e != cudaSuccess) { \
            printf("CUDA Error: \n");   \
            printf("    File    %s:\n", __FILE__);    \
            printf("    Line    %d:\n", __LINE__);    \
            printf("    Msg:    %s\n", cudaGetErrorString(e));  \
            exit(-1);   \
        }   \
    } while(0)

#define checkDriverError(call) {    \
    CUresult e = (call);   \
    if (e != CUDA_SUCCESS) { \
        const char* err_str;    \
        cuGetErrorString(e, &err_str);  \
        printf("CUDA Error: \n");   \
        printf("    File    %s:\n", __FILE__);    \
        printf("    Line    %d:\n", __LINE__);    \
        printf("    Msg:    %s\n", err_str);  \
        exit(-1);   \
    }   \
}

template<typename T>
T* cuda_malloc_wrapper(size_t size) {
    T* ptr;
    cudaMalloc((void**)&ptr, sizeof(T) * size);
    return ptr;
}

__host__ void create_tma_descriptor(
    CUtensorMap* desc,
    float* global_addr,
    int dim_m,
    int dim_n,
    int tile_m,
    int tile_n
) {
    uint64_t global_dims[] = {(uint64_t)dim_n, (uint64_t)dim_m};
    uint64_t global_stride[] = {(uint64_t)(dim_n * sizeof(float))};
    uint32_t box_dims[] = {(uint32_t)tile_n, (uint32_t)tile_m};
    uint32_t elem_stride[] = {1, 1};
    uint32_t rank = 2;

    checkDriverError(cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        global_addr,
        global_dims,
        global_stride,
        box_dims,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
}

#define M 128
#define N 128
#define TILE_M 32
#define TILE_N 32

__device__ void init_mbar(uint64_t* mbar, uint32_t count) {
    asm volatile (
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(count)
        : "memory"
    );
}

__device__ void set_expected_tx(uint64_t* mbar, uint32_t expected_tx) {
    asm volatile (
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(expected_tx)
        : "memory"
    );
}

__device__ void wait_tma(uint64_t* mbar, uint32_t phase) {
    asm volatile (
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_LOOP2:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_LOOP2;\n\t"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase)
        : "memory"
    );
}

__device__ void load_tma(float* shm_addr, const CUtensorMap* tensor_map, uint32_t cols, uint32_t rows, uint64_t* mbar) {
    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(shm_addr))
            "l"(tensor_map)
            "r"(cols)
            "r"(rows)
            "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
}

__device__ void fence_proxy() {
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ void load(float* shm_addr, float& val) {
    asm volatile (
        "ld.shared.f32 %0, [%1];\n"
        : "=f"(val)
        : "r"((uint32_t)__cvta_generic_to_shared(shm_addr))
    );
}

__device__ void store(float* global_addr, const float& val) {
    asm volatile (
        "st.global.f32 [%0], %1;\n"
        :: "l"(global_addr), "f"(val)
        : "memory"
    );
}

__global__ void matrix_add(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b,
    float* __restrict__ gl_c
) {
    
    extern __shared__ __align__(128) char shm[];
    float* shm_a = reinterpret_cast<float*>(shm);
    float* shm_b = reinterpret_cast<float*>(shm + TILE_M * TILE_N * sizeof(float));

    int row = blockIdx.y * TILE_M;
    int col = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    __shared__ __align__(8) uint64_t mbar;

    if (tid == 0) {
        init_mbar(&mbar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        set_expected_tx(&mbar, 2 * TILE_M * TILE_N * sizeof(float));

        load_tma(shm_a, &tensor_map_a, col, row, &mbar);
        load_tma(shm_b, &tensor_map_b, col, row, &mbar);

        wait_tma(&mbar, 0);
    }
    __syncthreads();       // 必须要加
    fence_proxy();

    for (int i = tid; i < TILE_M * TILE_N; i+=num_threads) {
        int local_row = i / TILE_N;
        int local_col = i % TILE_N;
        int global_row = row + local_row;
        int global_col = col + local_col;

        if (global_row < M && global_col < N) {
            float a, b, c;
            load(shm_a + i, a);
            load(shm_b + i, b);
            c = a + b;
            if (blockIdx.x == 0 && blockIdx.y && tid== 0) {
                printf("temp: %f\n", c);
            }
            gl_c[global_row * N + global_col] = c;
            // store(gl_c +  + global_row * N, c);
        }
    } 
}


int main() {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    printf("Device Compute Capability:  %d.%d\n", major, minor);

    if (major < 9) {
        printf("GPU Not Support TMA");
        return -1;
    }

    std::vector<float> h_a(M*N, 0.0f);
    std::vector<float> h_b(M*N, 0.0f);
    std::vector<float> h_c(M*N, 0.0f);
    std::vector<float> h_ref(M*N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);
    for (int i = 0; i < M * N; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
        h_ref[i] = h_a[i] + h_b[i];
    }

    float* d_a = cuda_malloc_wrapper<float>(M * N);
    float* d_b = cuda_malloc_wrapper<float>(M * N);
    float* d_c = cuda_malloc_wrapper<float>(M * N);

    cudaMemcpy(d_a, h_a.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);

    CUtensorMap tensor_map_a, tensor_map_b;
    create_tma_descriptor(&tensor_map_a, d_a, M, N, TILE_M, TILE_N);
    create_tma_descriptor(&tensor_map_b, d_b, M, N, TILE_M, TILE_N);

    dim3 block(16 * 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // launch kernel
    size_t shmem_size = 2 * TILE_M * TILE_N * sizeof(float);
    matrix_add<<<grid, block, shmem_size>>>(tensor_map_a, tensor_map_b, d_c);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(h_c.data(), d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_ref[i] - h_c[i]) > 1e-5) {
            if (errors < 10) {
                printf("Mismatch at %d, got %f, excepted %f\n", i, h_c[i], h_ref[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Success\n");
    } else {
        printf("FAILED\n");
    }
    return 0;
}
