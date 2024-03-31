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
    plan->precision = precision;

    switch (precision){
    case TCFFT_HALF:{
        switch (nx){
        case 256:
            plan->n_radices = 2;
            plan->radices = new int[2]{16, 16};
            plan->n_mergings = 1;
            plan->mergings = new int[1]{0};
            break;
        default:
            return TCFFT_NOT_SUPPORTED;
        }
        half* real_tmp = (half *)malloc(sizeof(half) * 256);
        half* imag_tmp = (half *)malloc(sizeof(half) * 256);
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                real_tmp[16 * i + j] = __float2half(cosf(2 * M_PI * i * j / 16));
                imag_tmp[16 * i + j] = __float2half(-sinf(2 * M_PI * i * j / 16));
            }
        }
        // 将旋转因子存入F_real和F_imag
        if (cudaMalloc(&plan->dft_real, sizeof(half) * 256) != cudaSuccess || cudaMalloc(&plan->dft_imag, sizeof(half) * 256) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        cudaMemcpy(plan->dft_real, real_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
        cudaMemcpy(plan->dft_imag, imag_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
        // 分配twiddle矩阵
        if (cudaMalloc(&plan->twiddle_real, sizeof(half) * 256) != cudaSuccess || cudaMalloc(&plan->twiddle_imag, sizeof(half) * 256) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                real_tmp[16 * i + j] = __float2half(cosf(2 * M_PI * i * j / 256));
                imag_tmp[16 * i + j] = __float2half(-sinf(2 * M_PI * i * j / 256));
            }
        }
        cudaMemcpy(plan->twiddle_real, real_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
        cudaMemcpy(plan->twiddle_imag, imag_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
        free(real_tmp);
        free(imag_tmp);
        break;
    }
    case TCFFT_SINGLE:
        break;
    case TCFFT_DOUBLE:
        break;
    default:
        return TCFFT_INVALID_TYPE;
    }
    return TCFFT_SUCCESS;
} 

/**
 * @brief 执行半精度一维FFT
*/
tcfftResult tcfftExecB2B(tcfftHandle plan, half *data){
    launch_half_256(data, plan);
    return TCFFT_SUCCESS;
}

/**
 * @brief 执行单精度一维FFT
*/
tcfftResult tcfftExecC2C(tcfftHandle plan, float *data){
    return TCFFT_SUCCESS;
}

tcfftResult tcfftDestroy(tcfftHandle plan){
    cudaFree(plan.dft_real);
    cudaFree(plan.dft_imag);
    cudaFree(plan.twiddle_real);
    cudaFree(plan.twiddle_imag);
    free(plan.radices);
    free(plan.mergings);
    return TCFFT_SUCCESS;
}