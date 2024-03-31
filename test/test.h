#include <cuda_runtime.h>
#include <fftw3.h>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
void setup(double *data, int n, int batch, int precision);
void doit(int iter);
void finalize(double *result);