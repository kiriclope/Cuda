SRC = GenerateConVecFile.cu GenConProbDistDepMat.cu Matrix_Utils.cu devHostConstants.h

a.out : GenerateConVecFile.cu GenConProbDistDepMat.cu Matrix_Utils.cu devHostConstants.h
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu

b.out : GenerateConVecFile.cu GenConProbDistDepMat.cu Matrix_Utils.cu devHostConstants.h
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o b.out

c.out : GenerateConVecFile.cu GenConProbDistDepMat.cu Matrix_Utils.cu devHostConstants.h
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o c.out

d.out : GenerateConVecFile.cu GenConProbDistDepMat.cu Matrix_Utils.cu devHostConstants.h
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o d.out

debug : GenerateConVecFile.cu
	nvcc -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.out

.PHONY: clean
