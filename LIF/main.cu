#include "cuda.h"
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "globalVars.h"
#include "aux.cu"
#include "cuda_histogram.h"

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)

void __cudaCheckLastError(const char* errorMessage, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError (msg, __FILE__, __LINE__)

void __cudaCheck(cudaError err, const char *file, const int line)
{
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
      file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

void __cudaCheckLastError(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

int main(int argc, char *argv[]) {

  for(t=0;t<nSteps;t++) { 
    nSpksInPrevStep = 0 ;
    devPtrs.t = t ; 
    histVecIndx = 0 ; 
    
    for(i=0; i<N_NEURONS;++i) 
      histCount[i] = 0 ;

    rkdumb<<<BlocksPerGrid, ThreadsPerBlock>>> (kernelParams, devPtrs) ; 
    cudaCheckLastError("rk") ; 

    if(t>0) { 
      cudaCheck(cudaMemcpyAsync(host_IF_SPK, dev_IF_SPK, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost, stream1)) ; 
      cudaCheck(cudaMemcpyAsync(host_Isyn, dev_Isyn,  N_NEURONS * sizeOfDbl, cudaMemcpyDeviceToHost, stream1)) ; 
    } 
    cudaCheck(cudaStreamSynchronize(stream1)) ; 

    for(i=0;i<N_NEURONS;++i) 
      if(host_IF_SPK[i]) { 
	if(t * DT > DISCARDTIME) 
	  Rates[i] += host_IF_SPK[i] ; 
	Spks[whichPop[i]] += 1 ; 
      }
        
    if(!(t%(int)(50.0/DT)))
      for(i=0;i<nbpop;i++) {
	fprintf(fpIFR, "%f ", ((double)Spks[i]) / (0.05 * (double)nbN[i])) ; fflush(fpIFR); 
	fprintf(stdout, "%f ", ((double)Spks[i]) / (0.05 * (double)nbN[i])) ; 
	Spks[i] = 0 ; 
      }
    fprintf(fpIFR,"\n") ; 
    fprintf(stdout,"\n") ; 

    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_histCount) ; 
    cudaCheckLastError("exp") ; 

    for(i=0;i<N_NEURONS;++i) 
      if(host_IF_SPK[i]){ 
	nSpksInPrevStep += 1;
        for(int jj = 0; jj < nPostNeurons[i]; ++jj) { 
          tmp = sparseConVec[idxVec[i] + jj] ; 
          histVec[histVecIndx++] = tmp ; 
        }
      }
    if(nSpksInPrevStep) { 
      cudaCheck(cudaMemcpy(dev_histVec, histVec, histVecIndx * sizeof(int), cudaMemcpyHostToDevice)) ; 
      callHistogramKernel<histogram_atomic_inc, 1>(dev_histVec, xform, sum, 0, histVecIndx, 0, &histCount[0], (int)N_NEURONS) ; 
      cudaCheckLastError("HIST") ; 
      cudaCheck(cudaMemcpyAsync(dev_histCount, histCount, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice, stream1)) ; 
    }    
    cudaCheck(cudaStreamSynchronize(stream1)) ; 
    computeConductanceHist<<<(N_NEURONS + 512 - 1) / 512, 512>>>(dev_histCountE, dev_histCountI) ; 
    cudaCheckLastError("g") ; 
    computeIsyn<<<BlocksPerGrid, ThreadsPerBlock>>>(t*DT, devPtrs) ; 
    cudaCheckLastError("isyp") ; 
  }

}