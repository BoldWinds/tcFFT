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

/**
 * @brief 支持批处理的256长度FFT计算
 * 
 * @param data      输入数据
 * @param result    输出数据
 * @param dft       DFT矩阵
 * @param twiddle   twiddle矩阵
*/
__global__ void single_256(float *data, float* result, float *dft, float *twiddle) {
    const int row = 16;
    __shared__ float temp[32][row];

    // 加载dft矩阵
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_2;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_2;
    wmma::load_matrix_sync(frag_dft_real_1, dft, 8);
    wmma::load_matrix_sync(frag_dft_real_2, dft + 128, 8);
    wmma::load_matrix_sync(frag_dft_imag_1, dft + 256, 8);
    wmma::load_matrix_sync(frag_dft_imag_2, dft + 384, 8);
    // 定义输入输出矩阵
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_real_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_real_2;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_imag_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::col_major> frag_data_imag_2;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_real;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_imag;

    // 读出指定位置的矩阵
    wmma::load_matrix_sync(frag_data_real_1, data + blockIdx.x * 256, 16);
    wmma::load_matrix_sync(frag_data_real_2, data + blockIdx.x * 256 + 8, 16);
    wmma::load_matrix_sync(frag_data_imag_1, data + blockIdx.x * 256 + gridDim.x * 256, 16);
    wmma::load_matrix_sync(frag_data_imag_2, data + blockIdx.x * 256 + gridDim.x * 256 + 8, 16);

    // 计算子序列的DFT结果矩阵
    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

    // 加载twiddle矩阵
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_twiddle_real;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_twiddle_imag;
    wmma::load_matrix_sync(frag_twiddle_real, twiddle, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_twiddle_imag, twiddle + 256, 16, wmma::mem_row_major);
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

    // 将计算结果重新存储回frag_data，然后在读取时转置
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_real_1_row;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_real_2_row;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_imag_1_row;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_imag_2_row;
    wmma::store_matrix_sync(reinterpret_cast<float*>(&temp[0][0]), frag_out_real, row, wmma::mem_col_major);
    wmma::store_matrix_sync(reinterpret_cast<float*>(&temp[16][0]), frag_out_imag, row, wmma::mem_col_major);
    wmma::load_matrix_sync(frag_data_real_1_row, reinterpret_cast<float*>(&temp[0][0]), row);
    wmma::load_matrix_sync(frag_data_real_2_row, reinterpret_cast<float*>(&temp[8][0]), row);
    wmma::load_matrix_sync(frag_data_imag_1_row, reinterpret_cast<float*>(&temp[16][0]), row);
    wmma::load_matrix_sync(frag_data_imag_2_row, reinterpret_cast<float*>(&temp[24][0]), row);

    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1_row, frag_data_real_2_row, frag_data_imag_1_row, frag_data_imag_2_row, frag_out_real, frag_out_imag);
    // 将数据存储回去
    wmma::store_matrix_sync(result + blockIdx.x * 256, frag_out_real, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(result + blockIdx.x * 256 + gridDim.x * 256, frag_out_imag, 16, wmma::mem_row_major);
}

__global__ void single_512(float *data, float* result, float *dft, float *twiddle) {
    extern __shared__ float smem[];

    // 加载dft矩阵
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_2;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_2;
    wmma::load_matrix_sync(frag_dft_real_1, dft, 8);
    wmma::load_matrix_sync(frag_dft_real_2, dft + 128, 8);
    wmma::load_matrix_sync(frag_dft_imag_1, dft + 256, 8);
    wmma::load_matrix_sync(frag_dft_imag_2, dft + 384, 8);
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
 * @brief 将16个长度为len的FFT合并为一个长度为16 * N的FFT
 * 
 * @param data      输入数据
 * @param dft       DFT矩阵
 * @param len       每个FFT的长度，用于确定读取矩阵时的步长
*/
__global__ void single_merge_16(float *data, float *dft){
    extern __shared__ float smem[];
    // 1. 计算按元素乘法
    //int len = blockDim.x * 16;
    //int nx = blockDim.x * blockDim.y * 256;
    //int batch = gridDim.z;
    //int stride = len;
    //int start = 16 * blockIdx.x + 16 * len * blockIdx.y;
    int row = threadIdx.x / 2;
    int between_real_imag = blockDim.x * blockDim.y * blockDim.z * 256;
    int thread_start = blockIdx.y * 256 * blockDim.x + blockIdx.x * 16 + row * blockDim.x * 16 + 8 * (threadIdx.x % 2);
    for (int i = 0; i < 8; i++){   
        double rad = 2 *  M_PI * row  * (16 * blockIdx.y + i + 8 * (threadIdx.x % 2)) / (16 * 16 * blockDim.x);
        smem[threadIdx.x * 8 + i] = data[thread_start + i] * cos(rad) - data[thread_start + i + between_real_imag] * sin(-rad);
        smem[threadIdx.x * 8 + i + 256] = data[thread_start + i] * sin(-rad) + data[thread_start + i + between_real_imag] * cos(rad);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_real_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_real_2;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_imag_1;
    wmma::fragment<wmma::matrix_b, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_data_imag_2;
    wmma::load_matrix_sync(frag_data_real_1, smem, 16);
    wmma::load_matrix_sync(frag_data_real_2, smem + 128, 16);
    wmma::load_matrix_sync(frag_data_imag_1, smem + 256, 16);
    wmma::load_matrix_sync(frag_data_imag_2, smem + 384, 16);
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_real_2;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_1;
    wmma::fragment<wmma::matrix_a, M_SINGLE, N_SINGLE, K_SINGLE, wmma::precision::tf32, wmma::row_major> frag_dft_imag_2;
    wmma::load_matrix_sync(frag_dft_real_1, dft, 8);
    wmma::load_matrix_sync(frag_dft_real_2, dft + 128, 8);
    wmma::load_matrix_sync(frag_dft_imag_1, dft + 256, 8);
    wmma::load_matrix_sync(frag_dft_imag_2, dft + 384, 8);
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_real;
    wmma::fragment<wmma::accumulator, M_SINGLE, N_SINGLE, K_SINGLE, float> frag_out_imag;
    
    // 2. 计算矩阵乘法
    complex_mul_single(frag_dft_real_1, frag_dft_real_2, frag_dft_imag_1, frag_dft_imag_2, frag_data_real_1, frag_data_real_2, frag_data_imag_1, frag_data_imag_2, frag_out_real, frag_out_imag);

    // 3. 同步数据
    wmma::store_matrix_sync(data + blockIdx.y * 256 * blockDim.x + blockIdx.x * 16, frag_out_real, blockDim.x * 16, wmma::mem_row_major);
    wmma::store_matrix_sync(data + blockIdx.y * 256 * blockDim.x + blockIdx.x * 16 + blockDim.x * blockDim.y * blockDim.z * 256, frag_out_imag, blockDim.x * 16, wmma::mem_row_major);
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

extern "C" tcfftResult launch_single(float *data, float *result, tcfftHandle plan) {
    single_256<<<plan.batch * plan.nx / 256, 32>>>(data, result, (float *)plan.dft, (float *)plan.twiddle);
    int len = 256;
    for (int i = 0; i < plan.n_mergings; i++)
    {   
        dim3 blocks = dim3(len / 16, plan.nx / (16 * len), plan.batch);
        single_merge_16<<<blocks, 32, sizeof(float) * 512>>>(result, (float *)plan.dft);
        len *= 16;
    }
    return TCFFT_SUCCESS;
}