SRC = GenerateConVecFile.cu

a.out : GenerateConVecFile.cu
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu
debug : GenerateConVecFile.cu
	nvcc -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.o *.out

.PHONY: clean
