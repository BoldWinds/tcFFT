#include "tcfft.h"
#include "tcfft_utils.h"
using namespace nvcuda;

__device__ void single_merge_2(float *data_out, float* data_in, int len);

__global__ void half_256(half *data, half *result, half *dft, half *twiddle) {
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
    wmma::store_matrix_sync(result + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(result + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
    
}

__global__ void single_256(float *data, float* result, float *dft, float *twiddle) {
    extern __shared__ float smem[];

    float *data_real = data;
    float *data_imag = data + 256;

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
    wmma::load_matrix_sync(frag_data_real_1, data_real, 16);
    wmma::load_matrix_sync(frag_data_real_2, data_real + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, data_imag, 16);
    wmma::load_matrix_sync(frag_data_imag_2, data_imag + 8, 16);


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
    wmma::store_matrix_sync(result + warp_start, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(result + warp_start + 256, frag_out_imag, 16, wmma::mem_row_major);
}

/**
 * @brief 支持批处理的256长度FFT计算
 * 
 * @param data      输入数据
 * @param result    输出数据
 * @param dft       DFT矩阵
 * @param twiddle   twiddle矩阵
 * @param num       一共计算的FFT数量
*/
__global__ void single_256_mul(float *data, float* result, float *dft, float *twiddle, int num) {
    extern __shared__ float smem[];

    // 目前，一个block有2个warp，计算两个256长度的FFT
    int block_start = blockIdx.x * 256 * 2;
    int warp_start = threadIdx.y * 256;

    float *data_real = data + block_start + warp_start;
    float *data_imag = data_real + num * 256;
    float *smem_real = smem + warp_start;
    float *smem_imag = smem_real + 2 * 256;
    float *result_real = result + block_start + warp_start;
    float *result_imag = result_real + num * 256;

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
    wmma::load_matrix_sync(frag_data_real_1, data_real, 16);
    wmma::load_matrix_sync(frag_data_real_2, data_real + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, data_imag, 16);
    wmma::load_matrix_sync(frag_data_imag_2, data_imag + 8, 16);


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
    wmma::store_matrix_sync(smem_real, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(smem_imag, frag_out_imag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_data_real_1, smem_real, 16);
    wmma::load_matrix_sync(frag_data_real_2, smem_real + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, smem_imag, 16);
    wmma::load_matrix_sync(frag_data_imag_2, smem_imag + 8, 16);

    // 重新计算
    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

    // 将数据存储回去
    wmma::store_matrix_sync(result_real, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(result_imag, frag_out_imag, 16, wmma::mem_row_major);
}

__global__ void single_512(float *data, float* result, float *dft, float *twiddle) {
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

    for (int i = 0; i < 2; i++)
    {
        wmma::load_matrix_sync(frag_data_real_1, data + warp_start, 16);
        wmma::load_matrix_sync(frag_data_real_2, data + warp_start + 8, 16);
        wmma::load_matrix_sync(frag_data_imag_1, data + warp_start + 512, 16);
        wmma::load_matrix_sync(frag_data_imag_2, data + warp_start + 520, 16);

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
        wmma::store_matrix_sync(smem + warp_start + 512, frag_out_imag, 16, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_data_real_1, smem + warp_start, 16);
        wmma::load_matrix_sync(frag_data_real_2, smem + warp_start + 8, 16);
        wmma::load_matrix_sync(frag_data_imag_1, smem + warp_start + 512, 16);
        wmma::load_matrix_sync(frag_data_imag_2, smem + warp_start + 520, 16);

        // 重新计算
        complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

        // 把结果存储到共享内存
        wmma::store_matrix_sync(smem + warp_start, frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(smem + warp_start + 512, frag_out_imag, 16, wmma::mem_row_major);
        warp_start += 256;
    }
    // 将数据存储回去
    single_merge_2(result, smem, 256);  
}

/**
 * @brief       单精度的两个FFT合并过程
 * 
 * @details     参与计算的两个FFT长度应均为len，后续也许可以修改
 * 
 * @param data_out
 * @param data_in
 * @param len
*/
__device__ void single_merge_2(float *data_out, float* data_in, int len){
    // 第一步，做按元素乘法，注意第一行和第一列可以跳过计算（乘1）
    int tid = threadIdx.x;
    int stride = len / 32;

    // 四个指针，分别指向第一个FFT和第二个FFT的实部和虚部
    float *real_1 = data_in;
    float *real_2 = data_in + len;
    float *imag_1 = data_in + 2*len;
    float *imag_2 = data_in + 3*len;

    // 只有两行，然而第一行不需要计算（W全是1）
    double a,b,c,d,real_w,imag_w;
    for (int i = 0; i < stride; i++)
    {
        real_w = cos(2 * M_PI * (tid * stride + i) / (2*len));
        imag_w = sin(-2 * M_PI * (tid * stride + i) / (2*len));
        a = real_2[tid * stride + i] * real_w;
        b = imag_2[tid * stride + i] * imag_w;
        c = real_2[tid * stride + i] * imag_w;
        d = imag_2[tid * stride + i] * real_w;
        real_2[tid * stride + + i] = a - b;
        imag_2[tid * stride + i] = c + d;
    }
    
    // 第二步，矩阵F2乘第一步的结果，利用F2的性质对矩阵乘法进行化简；每个warp中的thread，负责处理两个FFT的各8个元素
    for (int i = 0; i < stride; i++)
    {
        // 设目前正在处理乘法右侧矩阵的第k列
        // F2第一行(1,1)乘第k列
        data_out[0 + tid*stride + i] = real_1[tid*stride + i] + real_2[tid*stride + i];
        data_out[2*len + tid*stride + i] = imag_1[tid*stride + i] + imag_2[tid*stride + i];
        // F2第二行(1,-1)乘第k列
        data_out[len + tid*stride + i] = real_1[tid*stride + i] - real_2[tid*stride + i];
        data_out[3*len + tid*stride + i] = imag_1[tid*stride + i] - imag_2[tid*stride + i];
    }
}

extern "C" void launch_half_256(half* data, half* result,tcfftHandle plan) {
    // 调用CUDA核心 
    half_256<<<1, 32, sizeof(half) * plan.nx * 2>>>(data, result, (half *)plan.dft, (half *)plan.twiddle);
}

extern "C" void launch_single_256(float* data, float* result, tcfftHandle plan) {
    // 调用CUDA核心 
    if (plan.batch == 1)
    {
        single_256<<<1, 32, sizeof(float)*plan.nx*2>>>(data, result, (float *)plan.dft, (float *)plan.twiddle);
    }else{
        unsigned warp_per_block = 2;
        dim3 threads = {32 , warp_per_block};
        int blocks = plan.batch / warp_per_block;
        single_256_mul<<<blocks, threads, sizeof(float)*plan.nx*2*warp_per_block>>>(data, result, (float *)plan.dft, (float *)plan.twiddle, plan.batch);
    }
}

extern "C" void launch_single_512(float* data, float* result,tcfftHandle plan) {
    // 调用CUDA核心 
    single_512<<<1, 32, sizeof(float)*plan.nx*2>>>(data, result, (float *)plan.dft, (float *)plan.twiddle);
}