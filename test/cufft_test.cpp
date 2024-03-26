#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdlib>
#include <cuda_fp16.h>

half *data_host, *data_device;
cufftHandle plan;
int N, N_batch;

/**
 * @brief 设置FFT计算的长度、批次数量和输入数据，最后创建plan
 * 
 * @param data      输入数据，每个批次包含2*N个double类型数据（复数）
 * @param n         单次FFT的长度
 * @param batch   批次数量
*/
void setup(double *data, int n, int batch){
    N = n;                  // 设置单次FFT的长度
    N_batch = batch;        // 设置批次数量

    // 为主机上的数据分配内存，每个批次包含2*N个half类型数据（复数）
    data_host = (half *)malloc(sizeof(half) * N * N_batch * 2);
    // 将float类型的数据转换为half并复制到主机内存
    for (int i = 0; i < N_batch; ++i){
        for (int j = 0; j < N; ++j){
            data_host[(j + i * N) * 2 + 0] = __float2half((float)data[0 + j * 2 + i * N * 2]);
            data_host[(j + i * N) * 2 + 1] = __float2half((float)data[1 + j * 2 + i * N * 2]);
        }
    }

    cudaMalloc(&data_device, sizeof(half) * N * N_batch * 2);
    cudaMemcpy(data_device, data_host, sizeof(half) * N * N_batch * 2, cudaMemcpyHostToDevice);
    
    // 准备FFT计划
    long long p_n[1];       // FFT长度数组
    size_t worksize[1];     // 工作区大小数组
    p_n[0] = N;             // 设置FFT长度
    cufftCreate(&plan);     // 创建FFT计划
    // 创建多批次FFT计划，使用半精度浮点数
    cufftXtMakePlanMany(plan, 1, p_n,
                        NULL, 0, 0, CUDA_C_16F,
                        NULL, 0, 0, CUDA_C_16F, 
                        N_batch, worksize, CUDA_C_16F);
}

/**
 * @brief 执行多次FFT计算
 * 
 * @param iter 执行FFT的次数
*/
void doit(int iter){
    for (int i = 0; i < iter; ++i)
        cufftXtExec(plan, data_device, data_device, CUFFT_FORWARD);
    cudaDeviceSynchronize();
}

/**
 * @brief 将结果复制到host内存，并进行类型转换
 * 
 * @param result 输出结果，每个批次包含2*N个double类型数据（复数）
*/
void finalize(double *result){
    // 将结果从设备复制回主机
    cudaMemcpy(data_host, data_device, sizeof(half) * N * N_batch * 2, cudaMemcpyDeviceToHost);

    // 将half类型的结果转换回float类型
    for (int i = 0; i < N_batch; ++i){
        for (int j = 0; j < N; ++j){
            result[0 + j * 2 + i * N * 2] = __half2float(data_host[(j + i * N) * 2 + 0]);
            result[1 + j * 2 + i * N * 2] = __half2float(data_host[(j + i * N) * 2 + 1]);
        }
    }
}