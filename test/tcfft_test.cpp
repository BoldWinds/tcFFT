#include "../tcfft.h"
#include "test.h"
tcfftHandle plan;
void* in_device;

/**
 * @brief           转置函数
 * 
 * @details         递归转置函数，根据基数，对最小的子矩阵进行转置
 * 
 * @param n         数据长度
 * @param trans     转置后的数据
 * @param radices   转置的维度
 * @param n_radices 维度数
*/
void transpose(int n, int *trans, int *radices, int n_radices){
    if (n_radices == 2){
        // 直接进行转置
        int m = radices[0], n = radices[1];
        int *tmp = (int *)malloc(sizeof(int) * m * n);
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                tmp[j + n * i] = trans[j + n * i];
            }
        }
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                trans[j + m * i] = tmp[i + n * j];
            }
        }
    }else{
        int next_n = n / radices[n_radices-1];
        int step = radices[n_radices-1];
        for (int i = 0; i < step; i++){
            transpose(next_n, trans + i * next_n, radices, n_radices - 1);
        }
    }
}

/**
 * @brief           创建plan，计算索引，分配内存
 * 
 * @param data      输入数据
 * @param n         数据长度
 * @param n_batch   批次数
 * @param precision 精度, 1: float, 2: double, other: half
*/
void setup(double *data, int n, int batch, int precision){
    int *trans = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) trans[i] = i;
    switch (precision){
        case 1:{
            tcfftPlan1d(&plan, n, batch, TCFFT_SINGLE);
            transpose(n, trans, plan.radices, plan.n_radices);
            float* in_host = (float *)malloc(sizeof(float) * n * 2 * batch);
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = data[2 * n * j + 2 * trans[i] + 0];
                    in_host[n * j + i + n * batch] = data[2 * n * j + 2 * trans[i] + 1];
                }
            }
            cudaMalloc(&in_device, sizeof(float) * n * 2 * batch);
            cudaMemcpy(in_device, in_host, sizeof(float) * n * 2 * batch, cudaMemcpyHostToDevice);
            return;
        }
        case 2:{
            tcfftPlan1d(&plan, n, batch, TCFFT_DOUBLE);
            transpose(n, trans, plan.radices, plan.n_radices);
            double* in_host = (double *)malloc(sizeof(double) * n * 2 * batch);
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = data[2 * n * j + 2 * trans[i] + 0];
                    in_host[n * j + i + n * batch] = data[2 * n * j + 2 * trans[i] + 1];
                }
            }
            cudaMalloc(&in_device, sizeof(double) * n * 2 * batch);
            cudaMemcpy(in_device, in_host, sizeof(double) * n * 2 * batch, cudaMemcpyHostToDevice);
            return;
        }
        default:{
            tcfftPlan1d(&plan, n, batch, TCFFT_HALF);
            transpose(n, trans, plan.radices, plan.n_radices);
            half* in_host = (half *)malloc(sizeof(half) * n * 2 * batch);
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = __double2half(data[2 * n * j + 2 * trans[i] + 0]);
                    in_host[n * j + i + n * batch] = __double2half(data[2 * n * j + 2 * trans[i] + 1]);
                }
            }
            cudaMalloc(&in_device, sizeof(half) * n * 2 * batch);
            cudaMemcpy(in_device, in_host, sizeof(half) * n * 2 * batch, cudaMemcpyHostToDevice);
            return;
        }
    }
}

/**
 * @brief 将结果复制到host内存，并进行类型转换
 * 
 * @param result  结果数组
*/
void finalize(double *result){
    int n = plan.nx, batch = plan.batch;
    switch (plan.precision){
        case TCFFT_HALF:{
            half* in_host = (half *)malloc(sizeof(half) * n * 2 * batch);
            cudaMemcpy(in_host, in_device, sizeof(half) * n * 2 * batch, cudaMemcpyDeviceToHost);
            for (int j = 0; j < plan.batch; ++j){
                for (int i = 0; i < plan.nx; ++i){
                    result[2 * i + 2 * n * j] = in_host[i + n * j];
                    result[2 * i + 2 * n * j + 1] = in_host[i + n * j + n * batch];
                }
            }
            break;
        }
        case TCFFT_SINGLE:{
            float* in_host = (float *)malloc(sizeof(float) * n * 2 * batch);
            cudaMemcpy(in_host, in_device, sizeof(float) * n * 2 * batch, cudaMemcpyDeviceToHost);
            for (int j = 0; j < plan.batch; ++j){
                for (int i = 0; i < plan.nx; ++i){
                    result[2 * i + 2 * n * j] = in_host[i + n * j];
                    result[2 * i + 2 * n * j + 1] = in_host[i + n * j + n * batch];
                }
            }
            break;
        }
        case TCFFT_DOUBLE:{
            double* in_host = (double *)malloc(sizeof(double) * n * 2 * batch);
            cudaMemcpy(in_host, in_device, sizeof(double) * n * 2 * batch, cudaMemcpyDeviceToHost);
            for (int j = 0; j < plan.batch; ++j){
                for (int i = 0; i < plan.nx; ++i){
                    result[2 * i + 2 * n * j] = in_host[i + n * j];
                    result[2 * i + 2 * n * j + 1] = in_host[i + n * j + n * batch];
                }
            }
            break;
        }
        default:
            printf("error in precision\n");
            break;
    }
    tcfftDestroy(plan);
}

/**
 * @brief 执行iter次fft计算
 * 
 * @param iter  执行次数
*/
void doit(int iter){
    switch (plan.precision){
        case TCFFT_HALF:
            for (int t = 0; t < iter; ++t){
                tcfftExecB2B(plan, (half *)in_device);
            }
            break;
        case TCFFT_SINGLE:
            for (int t = 0; t < iter; ++t){
                tcfftExecC2C(plan, (float *)in_device);
            }
            break;
        case TCFFT_DOUBLE:
            break;
        default:
            printf("error in precision\n");
            break;
    }
    cudaDeviceSynchronize();
}