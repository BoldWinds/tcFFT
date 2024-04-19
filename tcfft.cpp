#include "tcfft.h"
#include <cuda_fp16.h>

tcfftResult setup_half(tcfftHandle *plan);
tcfftResult setup_single(tcfftHandle *plan);

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
    case TCFFT_HALF:
        return setup_half(plan);
    case TCFFT_SINGLE:
        return setup_single(plan);
    case TCFFT_DOUBLE:
        return TCFFT_NOT_SUPPORTED;
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
    return launch_single(data, result, plan);
}

tcfftResult tcfftDestroy(tcfftHandle plan){
    cudaFree(plan.dft);
    cudaFree(plan.twiddle);
    free(plan.radices);
    return TCFFT_SUCCESS;
}


tcfftResult setup_half(tcfftHandle *plan){
    switch (plan->nx){
        case 256:
            plan->n_radices = 2;
            plan->radices = new int[plan->n_radices]{16, 16};
            plan->n_mergings = 0;
            break;
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
    return TCFFT_SUCCESS;
}

tcfftResult setup_single(tcfftHandle *plan){
    switch (plan->nx){
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
        case 4096:
            plan->n_radices = 3;
            plan->radices = new int[plan->n_radices]{16, 16, 16};
            plan->n_mergings = 1;
            break;
        case 65536:
            plan->n_radices = 4;
            plan->radices = new int[plan->n_radices]{16, 16, 16, 16};
            plan->n_mergings = 2;
            break;
        default:
            return TCFFT_NOT_SUPPORTED;
    }
    float* tmp = (float *)malloc(sizeof(float) * 512);
    // 对dft矩阵进行重排，以适应wmma矩阵乘法的大小限制并减少多余global内存访问：将16*16行优先的矩阵变为两个16*8的行优先矩阵先后排列
    for (int i = 0; i < 16; i++){
        for(int j = 0; j < 8; j++){
            tmp[      8 * i + j] = cosf(2 * M_PI * i * j / 16);
            tmp[128 + 8 * i + j] = cosf(2 * M_PI * i * (j + 8) / 16);
            tmp[256 + 8 * i + j] = sinf(- 2 * M_PI * i * j / 16);
            tmp[384 + 8 * i + j] = sinf(- 2 * M_PI * i * (j + 8) / 16);
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
    // 分配merge时的twiddle矩阵

    return TCFFT_SUCCESS;
}