#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 256

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __constant__ uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

__device__ void sha256_u32(uint32_t input, uint32_t out[8]) {
    uint32_t w[64];
    w[0] = input;
    w[1] = 0x80000000u;
    #pragma unroll
    for (int i = 2; i < 14; ++i) {
        w[i] = 0u;
    }
    w[14] = 0u;
    w[15] = 32u;

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + s1 + ch + K[i] + w[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    out[0] = a + 0x6a09e667u;
    out[1] = b + 0xbb67ae85u;
    out[2] = c + 0x3c6ef372u;
    out[3] = d + 0xa54ff53au;
    out[4] = e + 0x510e527fu;
    out[5] = f + 0x9b05688cu;
    out[6] = g + 0x1f83d9abu;
    out[7] = h + 0x5be0cd19u;
}

__global__ void search_kernel(unsigned long long max_i, unsigned long long i_base,
                              unsigned long long stream_base, unsigned long long max_j,
                              uint32_t stream_len,
                              uint32_t *best_match, uint32_t *best_i, uint32_t *best_j) {
    __shared__ uint32_t tile[BLOCK_SIZE][8];
    __shared__ uint32_t sh_best[BLOCK_SIZE];
    __shared__ uint32_t sh_i[BLOCK_SIZE];
    __shared__ uint32_t sh_j[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    unsigned long long base_i = i_base + (unsigned long long)blockIdx.x * BLOCK_SIZE;
    unsigned long long i = base_i + tid;
    uint32_t tile_count = 0;
    if (base_i < max_i) {
        unsigned long long remaining = max_i - base_i;
        tile_count = remaining < BLOCK_SIZE ? (uint32_t)remaining : BLOCK_SIZE;
    }

    if (i < max_i) {
        sha256_u32((uint32_t)i, tile[tid]);
    } else {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            tile[tid][k] = 0u;
        }
    }
    __syncthreads();

    unsigned long long start_j = stream_base + tid;
    unsigned long long end_j = stream_base + stream_len;
    if (end_j > max_j) {
        end_j = max_j;
    }

    uint32_t local_best = 0u;
    uint32_t local_i = 0u;
    uint32_t local_j = 0u;

    if (start_j < end_j && tile_count > 0) {
        for (unsigned long long j = start_j; j < end_j; j += blockDim.x) {
            uint32_t h_j[8];
            uint32_t j32 = (uint32_t)j;
            sha256_u32(j32, h_j);
            for (uint32_t t = 0; t < tile_count; ++t) {
                if (j32 == (uint32_t)(base_i + t)) {
                    continue;
                }
                uint32_t diff = 0u;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    diff += __popc(h_j[k] ^ tile[t][k]);
                }
                uint32_t matches = 256u - diff;
                if (matches > local_best) {
                    local_best = matches;
                    local_i = (uint32_t)(base_i + t);
                    local_j = j32;
                }
            }
        }
    }

    sh_best[tid] = local_best;
    sh_i[tid] = local_i;
    sh_j[tid] = local_j;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sh_best[tid + stride] > sh_best[tid]) {
                sh_best[tid] = sh_best[tid + stride];
                sh_i[tid] = sh_i[tid + stride];
                sh_j[tid] = sh_j[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        uint32_t idx = blockIdx.x;
        best_match[idx] = sh_best[0];
        best_i[idx] = sh_i[0];
        best_j[idx] = sh_j[0];
    }
}

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void die_cuda(const char *msg, cudaError_t err) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
}

int main() {
    const uint32_t stream_len = 1u << 16;
    int device = 0;
    int blocks_per_sm = 0;
    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDevice(&device);
    if (err != cudaSuccess) die_cuda("cudaGetDevice failed", err);
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) die_cuda("cudaGetDeviceProperties failed", err);
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, search_kernel, BLOCK_SIZE, 0);
    if (err != cudaSuccess) die_cuda("cudaOccupancyMaxActiveBlocksPerMultiprocessor failed", err);
    if (blocks_per_sm <= 0) die("invalid occupancy calculation");

    uint32_t blocks = (uint32_t)(blocks_per_sm * prop.multiProcessorCount);
    if (blocks == 0) die("no active blocks available");
    uint32_t tile_size = blocks * BLOCK_SIZE;
    size_t buf_size = (size_t)blocks * sizeof(uint32_t);

    uint32_t *d_best_match = NULL;
    uint32_t *d_best_i = NULL;
    uint32_t *d_best_j = NULL;

    err = cudaMalloc(&d_best_match, buf_size);
    if (err != cudaSuccess) die_cuda("cudaMalloc failed for best_match", err);
    err = cudaMalloc(&d_best_i, buf_size);
    if (err != cudaSuccess) die_cuda("cudaMalloc failed for best_i", err);
    err = cudaMalloc(&d_best_j, buf_size);
    if (err != cudaSuccess) die_cuda("cudaMalloc failed for best_j", err);

    uint32_t *h_best_match = (uint32_t *)malloc(buf_size);
    uint32_t *h_best_i = (uint32_t *)malloc(buf_size);
    uint32_t *h_best_j = (uint32_t *)malloc(buf_size);
    if (!h_best_match || !h_best_i || !h_best_j) die("malloc failed");

    uint32_t best_match = 0u;
    uint32_t best_i = 0u;
    uint32_t best_j = 0u;

    printf("from hashlib import sha256\n");
    printf("def check(c, s, t):\n");
    printf("    a = sha256(bytes.fromhex(s)).digest()\n");
    printf("    b = sha256(bytes.fromhex(t)).digest()\n");
    printf("    assert c == sum((x ^ y ^ 0xff).bit_count() for (x, y) in zip(a, b))\n");

    auto start = std::chrono::high_resolution_clock::now();
    const unsigned long long max_j = 1ull << 32;
    const unsigned long long max_i_u64 = 1ull << 32;

    for (unsigned long long i_base = 0; i_base < max_i_u64; i_base += tile_size) {
        unsigned long long i_end = i_base + tile_size;
        if (i_end > max_i_u64) {
            i_end = max_i_u64;
        }

        unsigned long long stream_base = 0;
        while (stream_base < max_j) {
            search_kernel<<<blocks, BLOCK_SIZE>>>(i_end, i_base, stream_base, max_j,
                                                 stream_len, d_best_match, d_best_i, d_best_j);
            err = cudaGetLastError();
            if (err != cudaSuccess) die_cuda("kernel launch failed", err);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) die_cuda("cudaDeviceSynchronize failed", err);

            err = cudaMemcpy(h_best_match, d_best_match, buf_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) die_cuda("cudaMemcpy failed for best_match", err);
            err = cudaMemcpy(h_best_i, d_best_i, buf_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) die_cuda("cudaMemcpy failed for best_i", err);
            err = cudaMemcpy(h_best_j, d_best_j, buf_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) die_cuda("cudaMemcpy failed for best_j", err);

            uint32_t segment_best = 0u;
            uint32_t segment_i = 0u;
            uint32_t segment_j = 0u;

            for (uint32_t b = 0; b < blocks; ++b) {
                if (h_best_match[b] > segment_best) {
                    segment_best = h_best_match[b];
                    segment_i = h_best_i[b];
                    segment_j = h_best_j[b];
                }
            }

            if (segment_best > best_match) {
                best_match = segment_best;
                best_i = segment_i;
                best_j = segment_j;
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(now - start).count();
                printf("check(%u, \"%08x\", \"%08x\")  # found after %.6fs\n",
                       best_match, best_i, best_j, elapsed);
                fflush(stdout);
            }

            stream_base += stream_len;
        }
    }

    return 0;
}
