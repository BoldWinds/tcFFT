#include "test.h"
extern char *optarg;
extern int optopt;

int main(int argc, char *argv[]){
    // 默认参数
    int n = 65536, n_batch = 1, seed = 42, times = 1, precision = 0;
    char opt_c = 0;

    // 解析命令行参数
    while (EOF != (opt_c = getopt(argc, argv, "n:b:s:p:t:"))){
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
        case 'p':   // 设置随机种子
            precision = atoi(optarg);
            break;
        case 't':   // 设置随机种子
            times = atoi(optarg);
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

    // 使用CUFFT计算结果
    double *cu = (double *)malloc(sizeof(double) * n * n_batch * 2);
    cufft_get_result(data, cu, n, n_batch, precision, times);

    // 使用自定义的FFT实现计算测试结果
    double *tested = (double *)malloc(sizeof(double) * n * n_batch * 2);
    setup(data, n, n_batch, precision);
    doit(times);
    finalize(tested);

    // 计算并打印误差
    printf("cufft error rate: ");
    printf("%e\n", get_error(cu, standard, n, n_batch));
    printf("tcfft error rate: ");
    printf("%e\n", get_error(tested, standard, n, n_batch));

    /*printf("Test: \n");
    printMatrix(tested, 16, 16);
    printf("\n\n\n");
    printf("Standard: \n");
    printMatrix(standard, 16, 16)*/;
    return 0;
}
