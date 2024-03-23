#include "tcfft.h"
#include <cuda_fp16.h>

/**
 * @brief 创建一个FFT计划
 * 
 * @param plan      FFT计划
 * @param nx        FFT的长度
 * @param batch     批次数量
 * @param precision 精度
*/
tcfftResult tcfftPlan1d(tcfftHandle *plan, int nx, int batch, tcfftPrecision precision){
    if (plan == nullptr){
        return TCFFT_INVALID_PLAN;
    }
    if (nx <= 0 || batch <= 0){
        return TCFFT_INVALID_SIZE;
    }

    plan->nx = nx;
    plan->batch = batch;

    switch (nx){
    case 256:
        plan->n_radices = 3;
        plan->radices = new int[2]{16, 16};
        plan->n_mergings = 1;
        plan->mergings = new int[1]{0};
        break;
    default:
        return TCFFT_NOT_SUPPORTED;
    }

    switch (precision){
    case TCFFT_HALF:
        half* real_tmp = (half *)malloc(sizeof(half) * 256);
        half* imag_tmp = (half *)malloc(sizeof(half) * 256);
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                real_tmp[16 * i + j] = __float2half(cosf(2 * M_PI * i * j / 16));
                imag_tmp[16 * i + j] = __float2half(-sinf(2 * M_PI * i * j / 16));
            }
        }
        plan->F_real_host = real_tmp;
        plan->F_imag_host = imag_tmp;
        // 将旋转因子存入F_real和F_imag
        if (cudaMalloc(&plan->F_real, sizeof(half) * 256) != cudaSuccess || cudaMalloc(&plan->F_imag, sizeof(half) * 256) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        cudaMemcpy(plan->F_real, plan->F_real_host, sizeof(half) * 256, cudaMemcpyHostToDevice);
        cudaMemcpy(plan->F_imag, plan->F_imag_host, sizeof(half) * 256, cudaMemcpyHostToDevice);
        break;
    case TCFFT_SINGLE:
        break;
    case TCFFT_DOUBLE:
        break;
    default:
        return TCFFT_INVALID_TYPE;
    }
    return TCFFT_SUCCESS;
} 

tcfftResult tcfftExec(tcfftHandle plan, half *data){
    return TCFFT_SUCCESS;
}

tcfftResult tcfftDestroy(tcfftHandle plan){
    return TCFFT_SUCCESS;
}