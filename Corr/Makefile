SRC = main.cu CrossCorr.cu CudaFunc.cu ImportSpikes.cu GlobalVars.h

a.out : main.cu CrossCorr.cu CudaFunc.cu ImportSpikes.cu GlobalVars.h
	nvcc -arch=sm_35 -lcufft -O2 main.cu -lgsl -lgslcblas -lm

debug : main.cu CrossCorr1D.cu CudaFunc.cu ImportSpikes.cu GlobalVars.h
	nvcc -arch=sm_35 -lcufft -g -G -lineinfo $< -lgsl -lgslcblas -lm

clean:
	-rm *.out

.PHONY: clean
