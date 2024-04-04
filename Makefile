OBJ = test
	
FLAGS = -std=c++11 -lcublas -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -I/data/lbw/fftw/include -L/data/lbw/fftw/lib -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp

ifdef DEBUG
FLAGS += -g -G
endif

all : build/$(OBJ)

build/test : ./test/test.cpp ./test/test_utils.cpp ./test/tcfft_test.cpp tcfft.cpp tcfft.cu
	nvcc $^ -o $@ $(FLAGS)

.PHONY : clean

clean :
	rm -f build/$(OBJ)
