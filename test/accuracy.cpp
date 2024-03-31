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
            //data[    j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            //data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            data[    j * 2 + i * n * 2] = double(j);
            data[1 + j * 2 + i * n * 2] = 0;
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

    // 使用FFTW计算标准结果
    double *standard = (double *)malloc(sizeof(double) * n * n_batch * 2);
    fftw3_get_result(data, standard, n, n_batch);

    // 使用自定义的FFT实现计算测试结果
    double *tested = (double *)malloc(sizeof(double) * n * n_batch * 2);
    setup(data, n, n_batch, 0);
     doit(1);
    finalize(tested);

    // 计算并打印误差
    printf("%e\n", get_error(tested, standard, n, n_batch));

    printf("Test: \n");
    printMatrix(tested, 16, 16);
    printf("\n\n\n");
    printf("Standard: \n");
    printMatrix(standard, 16, 16);
    return 0;
}
