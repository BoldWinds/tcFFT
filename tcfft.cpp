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
            plan->radices = new int[plan->n_radices]{16, 16};
            plan->n_mergings = 0;
            break;
        case 512:
            plan->n_radices = 3;
            plan->radices = new int[plan->n_radices]{16, 16, 2};
            plan->n_mergings = 1;
            plan->mergings = new int[1]{0};
        case 1024:
            plan->n_radices = 3;
            plan->radices = new int[plan->n_radices]{16, 16, 4};
            plan->n_mergings = 1;
            plan->mergings = new int[1]{0};
        case 256*256:
            plan->n_radices = 4;
            plan->radices = new int[plan->n_radices]{16, 16, 16, 16};
            plan->n_mergings = 1;
            plan->mergings = new int[1]{0};
        default:
            return TCFFT_NOT_SUPPORTED;
        }
        half* tmp = (half *)malloc(sizeof(half) * 512);
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                tmp[16 * i + j] = __float2half(cosf(2 * M_PI * i * j / 16));
                tmp[16 * i + j + 256] = __float2half(-sinf(2 * M_PI * i * j / 16));
            }
        }
        // 将旋转因子存入F_real和F_imag
        if (cudaMalloc(&plan->dft, sizeof(half) * 512) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        cudaMemcpy(plan->dft, tmp, sizeof(half) * 512, cudaMemcpyHostToDevice);
        // 分配twiddle矩阵
        if (cudaMalloc(&plan->twiddle, sizeof(half) * 512) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                tmp[16 * i + j] = __float2half(cosf(2 * M_PI * i * j / 256));
                tmp[16 * i + j + 256] = __float2half(-sinf(2 * M_PI * i * j / 256));
            }
        }
        cudaMemcpy(plan->twiddle, tmp, sizeof(half) * 512, cudaMemcpyHostToDevice);
        free(tmp);
        break;
    }
    case TCFFT_SINGLE:{
        switch (nx){
            case 256:
                plan->n_radices = 2;
                plan->radices = new int[plan->n_radices]{16, 16};
                plan->n_mergings = 0;
                break;
            case 512:
                plan->n_radices = 3;
                plan->radices = new int[plan->n_radices]{16, 16, 2};
                plan->n_mergings = 0;
                break;
            default:
                return TCFFT_NOT_SUPPORTED;
        }
        float* tmp = (float *)malloc(sizeof(float) * 512);
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
                tmp[16 * i + j + 256] = -sinf(2 * M_PI * i * j / 16);
            }
        }
        // 将旋转因子存入dft
        if (cudaMalloc(&plan->dft, sizeof(float) * 512) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        cudaMemcpy(plan->dft, tmp, sizeof(float) * 512, cudaMemcpyHostToDevice);
        // 分配twiddle矩阵
        if (cudaMalloc(&plan->twiddle, sizeof(float) * 512) != cudaSuccess){
            return TCFFT_ALLOC_FAILED;
        }
        for (int i = 0; i < 16; ++i){
            for (int j = 0; j < 16; ++j){
                tmp[16 * i + j] = cosf(2 * M_PI * i * j / 256);
                tmp[16 * i + j + 256] = -sinf(2 * M_PI * i * j / 256);
            }
        }
        cudaMemcpy(plan->twiddle, tmp, sizeof(float) * 512, cudaMemcpyHostToDevice);
        free(tmp);
        break;
    }
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
tcfftResult tcfftExecB2B(tcfftHandle plan, half *data, half *result){
    launch_half_256(data, result, plan);
    return TCFFT_SUCCESS;
}

/**
 * @brief 执行单精度一维FFT
*/
tcfftResult tcfftExecC2C(tcfftHandle plan, float *data, float *result){
    switch (plan.nx)
    {
    case 256:
        launch_single_256(data, result, plan);
        break;
    case 512:
        launch_single_512(data, result, plan);
        break;
    default:
        return TCFFT_NOT_SUPPORTED;
    }
    return TCFFT_SUCCESS;
}

tcfftResult tcfftDestroy(tcfftHandle plan){
    cudaFree(plan.dft);
    cudaFree(plan.twiddle);
    free(plan.radices);
    free(plan.mergings);
    return TCFFT_SUCCESS;
}