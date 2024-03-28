#include "../tcfft.h"
#include "test.h"
int *rev, N, N_batch;
half *in_host, *in_device_0;
tcfftHandle plan;

/**
 * @brief           根据基数生成逆序索引，实际上是对每个FFT的输入矩阵进行转置
 * 
 * @param N         数据长度
 * @param rev       逆序索引
 * @param radices   基数
 * @param n_radices 基数数量
*/
/*void gen_rev(int N, int rev[], int radices[], int n_radices){
    int *tmp_0 = (int *)malloc(sizeof(int) * N);
    int *tmp_1 = (int *)malloc(sizeof(int) * N);
    int now_N = N;      // e.g 256， 之后在迭代中，会不断除以基数，相当于是对N进行分解
#pragma omp parallel for
    for (int i = 0; i < N; ++i)     tmp_0[i] = i;   // 为tmp_0赋初值，也就是输入数据的原始顺序
    for (int i = n_radices - 1; i >= 0; --i){   // 对于基数倒序遍历
#pragma omp parallel for
        for (int j = 0; j < N; j += now_N){     // 对于每个组
            for (int k = 0; k < radices[i]; ++k){   
                for (int l = 0; l < now_N / radices[i]; ++l){
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
            }
        }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        rev[i] = tmp_0[i];
}*/

/**
 * @brief 创建plan，计算索引，分配内存
 * 
 * @param data      输入数据
 * @param n         数据长度
 * @param n_batch   批次数
*/
void setup(double *data, int n, int batch){
    N = n;
    N_batch = batch;
    tcfftPlan1d(&plan, N, N_batch, TCFFT_HALF);
    int index = 0;
    in_host = (half *)malloc(sizeof(half) * N * 2 * N_batch);
    for (int i = 0; i < N_batch; i++){
        for(int j = 0; j < N; j++){
            index = 2 * j + i * N * 2;
            in_host[index] = __double2half(data[index]);
            in_host[index + 1] = __double2half(data[index + 1]);
        }
    }
    cudaMalloc(&in_device_0, sizeof(half) * N * 2 * N_batch);
    cudaMemcpy(in_device_0, in_host, sizeof(half) * N * 2 * N_batch, cudaMemcpyHostToDevice);
}

/**
 * @brief 将结果复制到host内存，并进行类型转换
 * 
 * @param result  结果数组
*/
void finalize(double *result){
    cudaMemcpy(in_host, in_device_0, sizeof(half) * N * 2 * N_batch, cudaMemcpyDeviceToHost);
#pragma omp paralllel for
    for (int j = 0; j < N_batch; ++j){
        for (int i = 0; i < N; ++i){
            result[0 + i * 2 + 2 * N * j] = in_host[2 * i + 0 + 2 * N * j];
            result[1 + i * 2 + 2 * N * j] = in_host[2 * i + 1 + 2 * N * j];
        }
    }
    tcfftDestroy(plan);
}

/**
 * @brief 执行iter次fft计算
 * 
 * @param iter  执行次数
*/
void doit(int iter){
    for (int t = 0; t < iter; ++t){
        tcfftExecB2B(plan, in_device_0);
    }
    cudaDeviceSynchronize();
}