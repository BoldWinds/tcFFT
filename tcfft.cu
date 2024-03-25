#include "tcfft.h"
#include "tcfft_utils.h"
using namespace nvcuda;

__global__ void layer_256_0(half2 *data, half *F_real, half *F_imag){
    extern __shared__ half2 smem_in[];
    // 计算线程块内的线程索引
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;   //test： 0-15
    // 根据线程块索引计算输入输出数据的起始位置
    int block_start = 0;
    //int block_start = blockIdx.x * 256 * CONT_SIZE;

    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, M_HALF, N_HALF, K_HALF, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);
    
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_data_real;
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_data_imag;
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_in_tmp0;
    wmma::fragment<wmma::matrix_b, M_HALF, N_HALF, K_HALF, half, wmma::col_major> frag_in_tmp1;
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_out_real;
    wmma::fragment<wmma::accumulator, M_HALF, N_HALF, K_HALF, half> frag_out_imag;

    // 读出指定位置的矩阵
    int warp_start = 0;
    //int warp_start = i + threadIdx.y * 256;
    wmma::load_matrix_sync(frag_in_tmp0, (half *)(data + block_start + warp_start), 32);
    wmma::load_matrix_sync(frag_in_tmp1, (half *)(data + block_start + warp_start) + 16, 32);

    // 将复数矩阵分别存储到实数和虚数矩阵
    // TODO 优化这一步
    int stride = 16 * threadIdx.x;
    for (int j = 0; j < 8; ++j){
        int index = j + stride;
        frag_data_real.x[index] = frag_in_tmp0.x[2 * index];
        frag_data_imag.x[index] = frag_in_tmp0.x[2 * index + 1];
        frag_data_real.x[8 + index] = frag_in_tmp1.x[2 * index];
        frag_data_imag.x[8 + index] = frag_in_tmp1.x[2 * index + 1];
    }

    complex_mul_half(frag_F_real, frag_F_imag, frag_data_real, frag_data_imag, frag_out_real, frag_out_imag);
    
    // 对于上一步矩阵乘法的结果进行转置
    wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_data_real, (half *)(smem_in + warp_start), 16);
    wmma::load_matrix_sync(frag_data_imag, (half *)(smem_in + warp_start) + 256, 16);
    
    // 进行element-wise乘法
    // TODO 优化这一步
    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col);
    half2 twiddle_factor = {1.0, 0};        // 第一行的旋转因子一定是1
    for (int j = 0; j < 16; ++j) {
        half2 in_ele = {frag_data_real.x[j], frag_data_imag.x[j]};
        in_ele = cmul(in_ele, twiddle_factor);
        frag_data_real.x[j] = in_ele.x;
        frag_data_imag.x[j] = in_ele.y;
        twiddle_factor = cmul(twiddle_factor, twiddle_unit);
    }

    complex_mul_half(frag_F_real, frag_F_imag, frag_data_real, frag_data_imag,  frag_out_real, frag_out_imag);
    // 将数据存储回去
    int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
    raw_col = threadIdx.x % 16 / 8 * 8;
    for (int j = 0; j < 8; ++j)
    {
        int row = raw_row;
        int col = j + raw_col;
        smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
    }
    __syncthreads();
}