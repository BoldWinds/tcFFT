#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "test.h"
extern char *optarg;
extern int optopt;

/**
 * @brief 计算测试结果与标准结果之间的误差
 * 
 * @param tested    测试结果指针
 * @param standard  标准结果指针
 * @param n         单次FFT的长度
 * @param n_batch   批次数量
*/
double get_error(double *tested, double *standard, int n, int n_batch){
    double error = 0;
    // 并行处理每个批次和每个点，计算误差
    for (int i = 0; i < n_batch; ++i){
        for (int j = 0; j < n; ++j){
            // 计算实部的误差
            double tested_e = tested[0 + j * 2 + i * n * 2];
            double standard_e = standard[0 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
            // 计算虚部的误差
            tested_e = tested[1 + j * 2 + i * n * 2];
            standard_e = standard[1 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
        }
    }
    // 返回平均误差
    return error / n / n_batch / 2;
}

/**
 * @brief 生成伪随机测试数据
 * 
 * @param data      输出数据，每个批次包含 2 * n * batch 个double类型数据（两个是为了复数）
 * @param n         单次FFT的长度
 * @param batch     批次数量
 * @param seed      随机数种子
*/
void generate_data(double *data, int n, int batch, int seed = 42){
    srand(seed);
    for (int i = 0; i < batch; ++i){
        for (int j = 0; j < n; ++j){
            data[0 + j * 2 + i * n * 2] = double(j);
            data[1 + j * 2 + i * n * 2] = 0.0;
            //data[0 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            //data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
        }
    }  
}


/**
 * @brief 用cufft计算标准结果
 * 
 * @param data      输入数据，每个批次包含 2 * n * batch 个double类型数据
 * @param standard  输出数据，每个批次包含 2 * n * batch 个double类型数据
 * @param n         单次FFT的长度
 * @param batch     批次数量
*/
void get_standard_result_half(double *data, double *standard, int n, int batch){
    half* data_host = (half *)malloc(sizeof(half) * n * batch * 2);

    for (int i = 0; i < batch; i++){
        for (int j = 0; j < n; j++){
            data_host[j * 2 + i * n * 2] = __float2half(data[j * 2 + i * n * 2]);
            data_host[j * 2 + 1 + i * n * 2] = __float2half(data[j * 2 + 1 + i * n * 2]);
        }   
    }

    half* data_device;
    cudaMalloc(&data_device, sizeof(half) * n * batch * 2);
    cudaMemcpy(data_device, data_host, sizeof(half) * n * batch * 2, cudaMemcpyHostToDevice);
    
    // 初始化CUFFT
    cufftHandle plan;
    cufftCreate(&plan);
    long long p_n[1] = {n};
    size_t worksize[1];
    cufftXtMakePlanMany(plan, 1, p_n,
                        NULL, 0, 0, CUDA_C_16F,
                        NULL, 0, 0, CUDA_C_16F, 
                        batch, worksize, CUDA_C_16F);

    // 执行FFT
    cufftXtExec(plan, data_device, data_device, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // 将结果转移到standard中
    cudaMemcpy(data_host, data_device, sizeof(half) * n * batch * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++){
        for(int j = 0; j < n; j++){
            standard[0 + j * 2 + i * n * 2] = __half2float(data_host[(j + i * n) * 2 + 0]);
            standard[1 + j * 2 + i * n * 2] = __half2float(data_host[(j + i * n) * 2 + 1]);
        }
    }
    
    // 销毁CUFFT
    cufftDestroy(plan);
    cudaFree(data_device);
    free(data_host);
}

/**
 * @brief 打印一个m行n列的复数矩阵
 * 
 * @param data  复数矩阵指针
 * @param m     行数
 * @param n     列数
*/
void printMatrix(double *data, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%f%+fi, ", data[j * 2 + i * n * 2], data[j * 2 + i * n * 2 + 1]);
        }
        printf("\n");
    }
}

// 程序入口
int main(int argc, char *argv[]){
    // 默认参数
    int n = 65536, n_batch = 1, seed = 42;
    char opt_c = 0;

    // 解析命令行参数
    while (EOF != (opt_c = getopt(argc, argv, "n:b:s:"))){
        switch (opt_c){
        case 'n':   // 设置FFT大小
            n = atoi(optarg);
            break;
        case 'b':   // 设置批次数
            n_batch = atoi(optarg);
            break;
        case 's':   // 设置随机种子
            seed = atoi(optarg);
            break;
        case '?':   // 未知选项
            printf("unknown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    // 分配内存并生成测试数据
    double *data = (double *)malloc(sizeof(double) * n * n_batch * 2);
    generate_data(data, n, n_batch, seed);

    // 使用CUFFT计算标准结果
    double *standard = (double *)malloc(sizeof(double) * n * n_batch * 2);
    get_standard_result_half(data, standard, n, n_batch);

    // 使用自定义的FFT实现计算测试结果
    double *tested = (double *)malloc(sizeof(double) * n * n_batch * 2);
    setup(data, n, n_batch);
    doit(1);
    finalize(tested);

    // 计算并打印误差
    printf("%e\n", get_error(tested, standard, n, n_batch));
    printf("%e\n", get_error(standard, standard, n, n_batch));

    printf("Data: \n");
    printMatrix(data, 16, 16);
    printf("Test: \n");
    printMatrix(tested, 16, 16);
    printf("Standard: \n");
    printMatrix(standard, 16, 16);

    return 0;
}
