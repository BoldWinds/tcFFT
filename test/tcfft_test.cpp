#include "../tcfft.h"
#include "test.h"
tcfftHandle plan;
void* in_device, *result_device;

/**
 * @brief 位逆序
 * 
 * @param N         长度
 * @param rev       存储位逆序后的索引数组
 * @param radices   基数数组
 * @param n_radices 基数数量
*/
void gen_rev(int N, int rev[], int radices[], int n_radices) {
    int *tmp_0 = (int *)malloc(sizeof(int) * N);
    int *tmp_1 = (int *)malloc(sizeof(int) * N);
    int now_N = N;

    // 初始化tmp_0数组
    for (int i = 0; i < N; ++i) tmp_0[i] = i;

    // 对于radices数组中的每个基数，执行重排操作
    for (int i = n_radices - 1; i >= 0; --i) {
        for (int j = 0; j < N; j += now_N) {
            for (int k = 0; k < radices[i]; ++k) {
                for (int l = 0; l < now_N / radices[i]; ++l) {
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
            }
        }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }

    // 将最终的逆序排列结果复制到rev
    for (int i = 0; i < N; ++i) rev[i] = tmp_0[i];

    // 释放临时数组
    free(tmp_0);
    free(tmp_1);
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
    switch (precision){
        case 1:{
            tcfftPlan1d(&plan, n, batch, TCFFT_SINGLE);
            gen_rev(n, trans, plan.radices, plan.n_radices);
            float* in_host = (float *)malloc(sizeof(float) * n * 2 * batch);            
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = data[2 * n * j + 2 * trans[i] + 0];
                    in_host[n * j + i + n * batch] = data[2 * n * j + 2 * trans[i] + 1];
                }
            }
            cudaMalloc(&in_device, sizeof(float) * n * 2 * batch);
            cudaMalloc(&result_device, sizeof(float) * n * 2 * batch);
            cudaMemcpy(in_device, in_host, sizeof(float) * n * 2 * batch, cudaMemcpyHostToDevice);
            return;
        }
        case 2:{
            tcfftPlan1d(&plan, n, batch, TCFFT_DOUBLE);
            gen_rev(n, trans, plan.radices, plan.n_radices);
            double* in_host = (double *)malloc(sizeof(double) * n * 2 * batch);
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = data[2 * n * j + 2 * trans[i] + 0];
                    in_host[n * j + i + n * batch] = data[2 * n * j + 2 * trans[i] + 1];
                }
            }
            cudaMalloc(&in_device, sizeof(double) * n * 2 * batch);
            cudaMalloc(&result_device, sizeof(double) * n * 2 * batch);
            cudaMemcpy(in_device, in_host, sizeof(double) * n * 2 * batch, cudaMemcpyHostToDevice);
            return;
        }
        default:{
            tcfftPlan1d(&plan, n, batch, TCFFT_HALF);
            gen_rev(n, trans, plan.radices, plan.n_radices);
            half* in_host = (half *)malloc(sizeof(half) * n * 2 * batch);
            for (int j = 0; j < batch; ++j){
                for (int i = 0; i < n; ++i){
                    in_host[n * j + i] = __double2half(data[2 * n * j + 2 * trans[i] + 0]);
                    in_host[n * j + i + n * batch] = __double2half(data[2 * n * j + 2 * trans[i] + 1]);
                }
            }
            cudaMalloc(&in_device, sizeof(half) * n * 2 * batch);
            cudaMalloc(&result_device, sizeof(half) * n * 2 * batch);
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
            cudaMemcpy(in_host, result_device, sizeof(half) * n * 2 * batch, cudaMemcpyDeviceToHost);
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
            cudaMemcpy(in_host, result_device, sizeof(float) * n * 2 * batch, cudaMemcpyDeviceToHost);
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
            cudaMemcpy(in_host, result_device, sizeof(double) * n * 2 * batch, cudaMemcpyDeviceToHost);
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
    tcfftExecC2C(plan, (float *)in_device, (float*) result_device);
    tcfftExecC2C(plan, (float *)in_device, (float*) result_device);
    tcfftExecC2C(plan, (float *)in_device, (float*) result_device);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    switch (plan.precision){
        case TCFFT_HALF:
            for (int t = 0; t < iter; ++t){
                tcfftExecB2B(plan, (half *)in_device, (half*) result_device);
                cudaDeviceSynchronize();
            }
            break;
        case TCFFT_SINGLE:
            for (int t = 0; t < iter; ++t){
                tcfftExecC2C(plan, (float *)in_device, (float*) result_device);
                cudaDeviceSynchronize();
            }
            break;
        case TCFFT_DOUBLE:
            break;
        default:
            printf("error in precision\n");
            break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Mine time: " << milliseconds << " ms" << std::endl;
}