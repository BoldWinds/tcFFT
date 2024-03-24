#ifndef TCFFT_UTILS_H
#define TCFFT_UTILS_H

#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

#define M_HALF 16
#define N_HALF 16
#define K_HALF 16
#define M_SINGLE 16
#define N_SINGLE 16
#define K_SINGLE 8
#define M_DOUBLE 8
#define N_DOUBLE 8
#define K_DOUBLE 4

using namespace std;
using namespace nvcuda;

/**
 * @brief 复数矩阵乘法, (A+Bi)(C+Di)
 * 
 * @param frag_F_real   矩阵A
 * @param frag_F_imag   矩阵B
 * @param frag_in_real  矩阵C
 * @param frag_in_imag  矩阵D
 * @param frag_out_real 输出的实数结果矩阵
 * @param frag_out_imag 输出的虚数结果矩阵
 */
__device__ inline void complex_mul_half(
    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> &frag_F_real,
    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> &frag_F_imag,
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> &frag_in_real,
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> &frag_in_imag,
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> &frag_out_real,
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> &frag_out_imag) {
    // 赋初值为0
    wmma::fill_fragment(frag_out_real, 0.0);
    wmma::fill_fragment(frag_out_imag, 0.0);

    // 设虚数分别为 a+bi  c+di
    // 矩阵B乘矩阵D，结果取反放入实矩阵
    wmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++)
        frag_out_real.x[i] = -frag_out_real.x[i];
    // 矩阵A乘矩阵C，结果放入实矩阵
    wmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);
    // 矩阵A乘矩阵D，结果放入虚矩阵
    wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    // 矩阵B乘矩阵C，结果放入虚矩阵
    wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

/**
 * @brief 计算半精度旋转因子
*/
__device__ __host__ inline half2 W_N_K(int N, int K){
    half2 t = {cosf(2 * M_PI * K / N), -sinf(2 * M_PI * K / N)};
    return t;
}

/**
 * @brief 复数乘法
*/
__device__ inline half2 const cmul(const half2 &a, const half2 &b){
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

/**
 * @brief 混合精度复数乘法
*/
__device__ inline half2 const cmul_mixed(const half2 &a, const float2 &b){
    return {a.x * __float2half(b.x) - a.y * __float2half(b.y), a.x * __float2half(b.y) + a.y * __float2half(b.x)};
}

__device__ inline void swap(half &a, half &b){
    half tmp = a;
    a = b;
    b = tmp;
}

#endif