#include "test.h"

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
            double tested_e = tested[2 * j + 2 * i * n];
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
 * @brief 使用FFTW库计算标准结果
 * 
 * @param data      输入数据指针
 * @param result    输出结果指针
 * @param n         单次FFT的长度
 * @param n_batch   批次数量
*/
void fftw3_get_result(double *data, double *result, int n, int n_batch){
    // 分配输入数组的内存
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_1d(n, in, in, FFTW_FORWARD, FFTW_ESTIMATE);

    // 循环处理每个批次
    for (int i = 0; i < n_batch; ++i){
        memcpy(in, data + 2 * i * n, sizeof(fftw_complex) * n);
        fftw_execute(p);
        memcpy(result + 2 * i * n, in, sizeof(fftw_complex) * n);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
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
            data[    j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
        }
    }  
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
        printf("\n\n");
    }
}

/**
 * @brief 将host上的data转换类型，并存放在device上
 * 
 * @param data      复数矩阵指针
 * @param n         单个FFT输入长度
 * @param batch     批处理数
 * 
 * @return          指向device上的data指针
*/
template <typename T>
T *get_device_data(double* data, int n, int batch){
    T *data_host, *device_data; // Change device_data declaration
    data_host = (T *)malloc(sizeof(T) * 2 * n * batch);
    if (sizeof(T) == sizeof(half)){
        for (int i = 0; i < batch; i++){
            for (int j = 0; j < n; j++){
                data_host[j * 2 + i * n * 2] = __double2half(data[j * 2 + i * n * 2]);
                data_host[j * 2 + i * n * 2 + 1] = __double2half(data[j * 2 + i * n * 2 + 1]);
            }
        }
    }else{
        for (int i = 0; i < batch; i++){
            for (int j = 0; j < n; j++){
                data_host[j * 2 + i * n * 2] = data[j * 2 + i * n * 2];
                data_host[j * 2 + i * n * 2 + 1] = data[j * 2 + i * n * 2 + 1];
            }
        }
    }
    
    cudaMalloc(&device_data, sizeof(T) * 2 * n * batch); // Corrected allocation
    cudaMemcpy(device_data, data_host, sizeof(T) * 2 * n * batch, cudaMemcpyHostToDevice);
    free(data_host);
    return device_data;
}

/**
 * @brief 将device上的result转换类型，并存放在host上
 * 
 * @param device_result device上的result指针
 * @param host_result   host上的result指针
 * @param n             单个FFT输入长度
 * @param batch         批处理数
*/
template <typename T>
void get_host_result(T *device_result, double* host_result, int n, int batch){
    T *host_tmp = (T*)malloc(sizeof(T) * 2 * n * batch);
    cudaMemcpy(host_tmp, device_result, sizeof(T) * 2 * n * batch, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++){
        for (int j = 0; j < n; j++){
            host_result[j * 2 + i * n * 2] = host_tmp[j * 2 + i * n * 2];
            host_result[j * 2 + i * n * 2 + 1] = host_tmp[j * 2 + i * n * 2 + 1];
        }
    }
    free(host_tmp);
}

/**
 * @brief 将device上的result转换类型，并存放在host上
 * 
 * @param data      host上的输入data指针
 * @param result    host上的输出result指针
 * @param n         单个FFT输入长度
 * @param batch     批处理数
 * @param times     执行次数
*/
template <typename T>
void cufft_exec(double *data, double *result, int n, int batch, int times){
    cufftHandle plan;
    long long p_n[1] = {n}; // FFT长度数组
    size_t worksize[1];     // 工作区大小数组
    cufftCreate(&plan);     // 创建FFT计划
    switch (sizeof(T))
    {
    case sizeof(half):
        cufftXtMakePlanMany(plan, 1, p_n,
                    NULL, 0, 0, CUDA_C_16F,
                    NULL, 0, 0, CUDA_C_16F, 
                    batch, worksize, CUDA_C_16F);
        break;
    case sizeof(float):
        cufftXtMakePlanMany(plan, 1, p_n,
                    NULL, 0, 0, CUDA_C_32F,
                    NULL, 0, 0, CUDA_C_32F, 
                    batch, worksize, CUDA_C_32F);
        break;
    
    case sizeof(double):
        cufftXtMakePlanMany(plan, 1, p_n,
                    NULL, 0, 0, CUDA_C_64F,
                    NULL, 0, 0, CUDA_C_64F, 
                    batch, worksize, CUDA_C_64F);
        break;
    default:
        printf("error in plan make!\n");
        break;
    }
    T* device_data = get_device_data<T>(data, n, batch);
    T* device_result;
    cudaMalloc(&device_result, sizeof(T) * 2 * n * batch);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++){
        cufftXtExec(plan, device_data, device_result, CUFFT_FORWARD);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    get_host_result<T>(device_result, result, n , batch);
    printf("CUFFT use time: %lf ms\n", duration.count());
}


/**
 * @brief 根据输入的精度，调用不同类型的cufft_exec
 * 
 * @param data      host上的输入data指针
 * @param result    host上的输出result指针
 * @param n         单个FFT输入长度
 * @param batch     批处理数
 * @param times     执行次数
 * @param precision 精度选择
*/
void cufft_get_result(double* data, double *result, int n, int batch, int precision, int times) {
    switch (precision)
    {
    case 0:{
        cufft_exec<half>(data,result,n,batch,times);
        break;
    }
    case 1:{
        cufft_exec<float>(data,result,n,batch,times);
        break;
    }
    case 2:{
        cufft_exec<double>(data,result,n,batch,times);
        break;
    }
    default:
        printf("error in cufft!\n");
        break;
    }
}