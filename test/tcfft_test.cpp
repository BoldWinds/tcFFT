#include "../tcfft.h"
#include "test.h"
tcfftHandle plan;
void* in_device, *result_device;

/**
 * @brief           转置函数
 * 
 * @param trans     转置矩阵指针
 * @param row       行数
 * @param col       列数
*/
void transpose(int *trans, int row, int col){
    int *tmp = (int *)malloc(sizeof(int) * row * col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            tmp[j + col * i] = trans[j + col * i];
        }
    }
    for (int i = 0; i < col; i++){
        for (int j = 0; j < row; j++){
            trans[j + row * i] = tmp[i + col * j];
        }
    }
    free(tmp);
}

/**
 * @brief           数据重排列函数
 * 
 * @param n         数据长度
 * @param data     要处理的数据
 * @param radices   分解的基数
 * @param n_radices 基数数量
*/
void reposition(int n, int *data, int *radices, int n_radices){
    if (n_radices == 2){
        transpose(data, radices[1], radices[0]);
    }else{
        int row = radices[n_radices - 1];
        int col = n / radices[n_radices - 1];
        transpose(data, col, row);
        for (int i = 0; i < row; i++){
            reposition(col, data + i * col, radices, n_radices - 1);
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
            reposition(n, trans, plan.radices, plan.n_radices);
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
            reposition(n, trans, plan.radices, plan.n_radices);
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
            reposition(n, trans, plan.radices, plan.n_radices);
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    switch (plan.precision){
        case TCFFT_HALF:
            for (int t = 0; t < iter; ++t){
                tcfftExecB2B(plan, (half *)in_device, (half*) result_device);
            }
            break;
        case TCFFT_SINGLE:
            for (int t = 0; t < iter; ++t){
                tcfftExecC2C(plan, (float *)in_device, (float*) result_device);
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