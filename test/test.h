#ifndef TEST_H
#define TEST_H 
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
double get_error(double *tested, double *standard, int n, int n_batch);
void fftw3_get_result(double *data, double *result, int n, int n_batch);
void generate_data(double *data, int n, int batch, int seed);
void printMatrix(double *data, int m, int n);
template <typename T>
void cufft_exec(double *data, double *result, int n, int batch, int times);
void cufft_get_result(double* data, double *result, int n, int batch, int precision, int times);
#endif