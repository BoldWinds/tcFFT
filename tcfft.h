#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

typedef enum tcfftResult_t {
  TCFFT_SUCCESS        = 0x0,
  TCFFT_INVALID_PLAN   = 0x1,
  TCFFT_ALLOC_FAILED   = 0x2,
  TCFFT_INVALID_TYPE   = 0x3,
  TCFFT_INVALID_VALUE  = 0x4,
  TCFFT_INTERNAL_ERROR = 0x5,
  TCFFT_EXEC_FAILED    = 0x6,
  TCFFT_SETUP_FAILED   = 0x7,
  TCFFT_INVALID_SIZE   = 0x8,
  TCFFT_NOT_SUPPORTED  = 0x9,

} tcfftResult;

typedef enum tcfftPrecision_t {
  TCFFT_HALF    = 0x10,
  TCFFT_SINGLE  = 0x11,
  TCFFT_DOUBLE  = 0x12,
} tcfftPrecision;

struct tcfftHandle{
    int nx;           // FFT的长度
    int batch;        // 批次数量
    int n_radices;    // 基数数量
    int* radices;
    //int radices[9] = {16, 16, 16, 16, 16, 16, 16, 16, 16};  // 基数数组
    int n_mergings;     // 合并数量
    int* mergings;
    //int mergings[3] = {0, 0, 0};    // 合并数组
    //void (*layer_0[3])(half2 *, half *, half *);
    //void (*layer_1[3])(int, half2 *, half *, half *);
    void *F_real, *F_imag;                  // 位于device
    void *F_real_host, *F_imag_host;        // 位于host
};

tcfftResult tcfftCreate(tcfftHandle *plan);

tcfftResult tcfftPlan1d(tcfftHandle *plan, int nx, int batch);

tcfftResult tcfftExecB2B(tcfftHandle plan, half *data);

tcfftResult tcfftExecC2C(tcfftHandle plan, half *data);

tcfftResult tcfftDestroy(tcfftHandle plan);
