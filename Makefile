FLAGS=-O3 -std=c++11 -arch=sm_35 -ftz=true #-Xcompiler -rdynamic -lineinfo -g
DEPS = bfs.cu scan.cuh fileIO.h utils.h

all:bfs.cu $(DEPS)
	nvcc ${FLAGS} $< -o $@

clean:
	rm -rf *.o all



