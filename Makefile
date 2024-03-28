OBJ = accuracy
	
FLAGS = -std=c++11 -lcublas -gencode arch=compute_86,code=sm_86 -res-usage -lcudart -lcufft -lineinfo -Xcompiler -fopenmp

ifdef DEBUG
FLAGS += -g -G
endif

all : build/$(OBJ)

build/accuracy : ./test/accuracy.cpp ./test/tcfft_test.cpp tcfft.cpp tcfft.cu
	nvcc $^ -o $@ $(FLAGS)

.PHONY : clean

clean :
	rm -f build/$(OBJ)
