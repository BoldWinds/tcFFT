#include "tcfft.h"
#include "tcfft_utils.h"
using namespace nvcuda;

__global__ void half_256(half *data, half *dft, half *twiddle) {
    extern __shared__ half smem_in[];

    // 加载dft矩阵
    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> frag_dft_real;
    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> frag_dft_imag;
    wmma::load_matrix_sync(frag_dft_real, dft, 16);
    wmma::load_matrix_sync(frag_dft_imag, dft + 256, 16);
    // 加载twiddle矩阵
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_twiddle_real;
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_twiddle_imag;
    wmma::load_matrix_sync(frag_twiddle_real, twiddle, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_twiddle_imag, twiddle + 256, 16, wmma::mem_row_major);
    // 定义输入输出矩阵
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_data_real;
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_data_imag;
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_out_real;
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_out_imag;

    // 读出指定位置的矩阵
    int warp_start = 0;
    wmma::load_matrix_sync(frag_data_real, data, 16);
    wmma::load_matrix_sync(frag_data_imag, data + 256, 16);

    // 计算子序列的DFT结果矩阵
    complex_mul_half(frag_dft_real, frag_dft_imag, frag_data_real, frag_data_imag, frag_out_real, frag_out_imag);

    // 按元素乘twiddle矩阵
    double a, b, c, d;
    for (int i = 0; i < frag_out_real.num_elements; i++){
        a = frag_out_real.x[i] * frag_twiddle_real.x[i];
        b = frag_out_imag.x[i] * frag_twiddle_imag.x[i];
        c = frag_out_real.x[i] * frag_twiddle_imag.x[i];
        d = frag_out_imag.x[i] * frag_twiddle_real.x[i];
        frag_out_real.x[i] = __double2half(a-b);
        frag_out_imag.x[i] = __double2half(c+d);
    }
    __syncthreads();
    // 将计算结果转置并重新存储回frag_data
    wmma::store_matrix_sync(smem_in + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(smem_in + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_data_real, smem_in + warp_start, 16);
    wmma::load_matrix_sync(frag_data_imag, smem_in + warp_start + 256, 16);

    // 重新计算
    complex_mul_half(frag_dft_real, frag_dft_imag, frag_data_real, frag_data_imag,  frag_out_real, frag_out_imag);

    // 将数据存储回去
    wmma::store_matrix_sync(data + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(data + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
    
}

__global__ void single_256(float *data, float *dft, float *twiddle) {
    extern __shared__ float smem[];

    // 加载dft矩阵
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_2;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_2;
    wmma::load_matrix_sync(frag_dft_real_1, dft, 16);
    wmma::load_matrix_sync(frag_dft_real_2, dft + 8, 16);
    wmma::load_matrix_sync(frag_dft_imag_1, dft + 256, 16);
    wmma::load_matrix_sync(frag_dft_imag_2, dft + 264, 16);
    // 加载twiddle矩阵
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_twiddle_real;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_twiddle_imag;
    wmma::load_matrix_sync(frag_twiddle_real, twiddle, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_twiddle_imag, twiddle + 256, 16, wmma::mem_row_major);
    // 定义输入输出矩阵
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_real_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_real_2;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_imag_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_imag_2;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_real;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_imag;

    // 读出指定位置的矩阵
    int warp_start = 0;
    wmma::load_matrix_sync(frag_data_real_1, data, 16);
    wmma::load_matrix_sync(frag_data_real_2, data + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, data + 256, 16);
    wmma::load_matrix_sync(frag_data_imag_2, data + 264, 16);


    // 计算子序列的DFT结果矩阵
    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

    // 按元素乘twiddle矩阵
    double a, b, c, d;
    for (int i = 0; i < frag_out_real.num_elements; i++){
        a = frag_out_real.x[i] * frag_twiddle_real.x[i];
        b = frag_out_imag.x[i] * frag_twiddle_imag.x[i];
        c = frag_out_real.x[i] * frag_twiddle_imag.x[i];
        d = frag_out_imag.x[i] * frag_twiddle_real.x[i];
        frag_out_real.x[i] = float(a - b);
        frag_out_imag.x[i] = float(c + d);
    }
    __syncthreads();
    // 将计算结果转置并重新存储回frag_data
    wmma::store_matrix_sync(smem + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(smem + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_data_real_1, smem, 16);
    wmma::load_matrix_sync(frag_data_real_2, smem + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, smem + 256, 16);
    wmma::load_matrix_sync(frag_data_imag_2, smem + 264, 16);

    // 重新计算
    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

    // 将数据存储回去
    wmma::store_matrix_sync(data + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(data + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
}

extern "C" void launch_half_256(half* data, tcfftHandle plan) {
    // 调用CUDA核心 
    half_256<<<1, 32, sizeof(half) * plan.nx * 2 * plan.batch>>>(data, (half *)plan.dft, (half *)plan.twiddle);
}

extern "C" void launch_single_256(float* data, tcfftHandle plan) {
    // 调用CUDA核心 
    single_256<<<1, 32, sizeof(float)*plan.nx*2*plan.batch>>>(data, (float *)plan.dft, (float *)plan.twiddle);
}