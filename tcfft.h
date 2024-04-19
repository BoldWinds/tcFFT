#ifndef TCFFT_H
#define TCFFT_H
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
  TCFFT_UNDEFINED      = 0xA,
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
    int n_mergings;
    tcfftPrecision precision;
    void *dft;
    void *twiddle;
    void *merge_twiddles;
};

tcfftResult tcfftPlan1d(tcfftHandle *plan, int nx, int batch, tcfftPrecision precision);

tcfftResult tcfftExecB2B(tcfftHandle plan, half *data, half *result);

tcfftResult tcfftExecC2C(tcfftHandle plan, float *data, float *result);

tcfftResult tcfftDestroy(tcfftHandle plan);

// 以下是内核函数的相关定义
extern "C" tcfftResult launch_single(float* data, float* result, tcfftHandle plan);
extern "C" void launch_half_256(half* data, half* result,tcfftHandle plan);
#endif