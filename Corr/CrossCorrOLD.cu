#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include "librairies.h"
#include "CudaFunc.cu"
#include "cuPrintf.cu"

#include "ImportSpikes.cu"
#include "CrossCorr.cu"

#define Nx 2620UL
#define Ny 2620UL
#define BATCH 1

#define N_THREADS 256

///////////////////////////////////////////////

__host__ __device__ cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
  
  cufftComplex c;
  c.x = a.x * b.x - a.y * b.y;  
  c.y = a.x * b.y + a.y * b.x;  

  return c ;
}

///////////////////////////////////////////////

__host__ float Norm(float* x, unsigned long Size) {
  unsigned long i ;
  float y=0 ;
  for(i=0;i<Size;i++)
    y += x[i]*x[i] ;

  return sqrt(y) ;

}
///////////////////////////////////////////////

__global__ void cudaMultiply(cufftComplex *dev_xdata, cufftComplex *dev_ydata, cufftComplex *dev_zdata, unsigned long dataSize) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  cufftComplex c;  
  
  if( id < dataSize ) {
    c = (cufftComplex) cuConjf( (cuComplex) dev_ydata[id] ) ;
    dev_zdata[id] = (cufftComplex) ComplexMul( dev_xdata[id], c ) ;
    cuPrintf("id %d zdata %f +i%f\r", id, dev_zdata[id].x, dev_zdata[id].y) ;
  }
  
}

///////////////////////////////////////////////

int main(int argc, char *argv[]) {

  cudaPrintfInit();

  ///////////////////////////
  // Import Data
  ///////////////////////////

  printf("Import Data ") ;
  
  // float *data ;
  // data = (float *) malloc( (unsigned long) Nx * Ny * sizeof(float) * BATCH );
  
  // FILE *file;
  // file = fopen("../../MATLAB/sensorData.dat","rb") ;

  // int dum ;
  // dum = fread(&data[0], sizeof(float), (unsigned long) Nx*Ny*BATCH, file) ;
  // fclose(file) ;

  // for(int i=0;i<10;i++) 
  //   printf("%f %f | ", data[i], data[i+Nx] ) ;
  // printf("\n") ;

  // unsigned long Nfft = pow( 2, ( ceil( log2( (float) ( Nx+Ny-1 ) ) ) ) ) ; 

  // printf("Nx %lu Ny %lu Nfft %lu ", Nx, Ny, Nfft) ;

  // float *xdata,*ydata ;
  // xdata = (float *) malloc( (unsigned long) Nfft * sizeof(float) * BATCH );
  // ydata = (float *) malloc( (unsigned long) Nfft * sizeof(float) * BATCH );

  // for(unsigned long i=0;i<Nfft;i++) {

  //   if(i<Nx)
  //     // xdata[i] = cos( (float) i );
  //     xdata[i] = data[i] ;
  //   else
  //     xdata[i] = 0. ;

  //   if(i<Ny)
  //     // ydata[i] = sin( (float) i );
  //   ydata[i] = data[i+Nx] ; 
  //   else
  //     ydata[i] = 0. ;    
  // }

  unsigned long Nfft ; 
  float **SpkTimes ;
  ImportSpikeTrains(Nfft, SpkTimes) ;
  
  ///////////////////////////
  // cufft utils
  ///////////////////////////

  printf("... Done\nAllocate device pointers ") ;
  
  cufftReal *dev_xdata, *dev_ydata, *dev_xyCorr ;
  cufftComplex *dev_iXdata, *dev_iYdata, *dev_iZdata ;

  cudaCheck( cudaMalloc((void**)&dev_xdata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_ydata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_xyCorr, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;

  cudaCheck( cudaMalloc((void**)&dev_iXdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iYdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iZdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  
  float *xyCorr= NULL ;
  xyCorr = (float *) malloc( (unsigned long) Nfft * sizeof(float) );
  
  printf("... Done\nCopy Host to Dev") ;
  
  cudaCheck( cudaMemcpy(dev_xdata, xdata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH, cudaMemcpyHostToDevice) ) ; 
  cudaCheck( cudaMemcpy(dev_ydata, ydata, (unsigned long) Nfft * sizeof(cufftReal)* BATCH, cudaMemcpyHostToDevice) ) ; 

  ///////////////////////////
  
  cufftHandle plan ;

  if (cufftPlan1d(&plan, Nfft, CUFFT_R2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    exit(-1) ;
  }
  
  ///////////////////////////

  printf("... Done\nFirst FFT ") ;
  /* Use the CUFFT plan to transform the signal in place. */
  if (cufftExecR2C(plan, dev_xdata, dev_iXdata) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
    exit(-1) ;
  }

  printf("... Done\n Second FFT ") ;
  if (cufftExecR2C(plan, dev_ydata, dev_iYdata) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecR2C Forward failed\n");
    exit(-1) ;
  }
  
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  ///////////////////////////////////////
  // Kernel Setup for multiplication
  ///////////////////////////////////////

  printf("... Done\n  Multiply FFT ") ;

  ///////////
  // on CPU
  ///////////

  // cufftComplex *iXdata, *iYdata, *iZdata ;
  // iXdata = (cufftComplex *) malloc( (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) );
  // iYdata = (cufftComplex *) malloc( (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) );
  // iZdata = (cufftComplex *) malloc( (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) );

  // cudaCheck( cudaMemcpy(iXdata, dev_iXdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost) ) ; 
  // cudaCheck( cudaMemcpy(iYdata, dev_iYdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost) ) ; 

  // for(unsigned long i=0;i<Nfft/2+1;i++)
  //   iZdata[i] = (cufftComplex) ComplexMul( iXdata[i], cuConjf( iYdata[i] ) ) ;
    
  // cudaCheck( cudaMemcpy(dev_iZdata, iZdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex), cudaMemcpyHostToDevice) ) ; 

  // for(int i=0;i<2;i++)
  //   printf("%f +i%f * %f +i%f = %f +i%f \n", iXdata[i].x, iXdata[i].y, iYdata[i].x, iYdata[i].y, iZdata[i].x, iZdata[i].y) ;

  ///////////
  // on GPU
  ///////////
  
  int ThreadsPerBlock = N_THREADS ;
  int BlocksPerGrid = ( Nfft + ThreadsPerBlock-1 ) / ThreadsPerBlock;

  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }

  cudaMultiply<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_iXdata, dev_iYdata, dev_iZdata, (unsigned long) Nfft/2+1 ) ; 
  cudaPrintfDisplay(stdout, true);
  
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  ///////////////////////////

  printf("... Done\n   Invert FFT ") ;
  
  if (cufftPlan1d(&plan, Nfft, CUFFT_C2R, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    exit(-1);
  }

  if (cufftExecC2R(plan, dev_iZdata, dev_xyCorr) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2R Forward failed\n");
    exit(-1) ;
  }

  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  cufftDestroy(plan) ;

  ///////////////////////////
  
  printf("... Done\nFree Memory \n") ;
  
  free(data) ;
  free(xdata) ;
  free(ydata) ;
  
  cudaFree(dev_xdata) ; 
  cudaFree(dev_ydata) ;

  cudaFree(dev_iXdata) ;
  cudaFree(dev_iYdata) ;
  cudaFree(dev_iZdata) ;

  ///////////////////////////
  
  cudaCheck( cudaMemcpy(xyCorr, dev_xyCorr, (unsigned long) Nfft * sizeof(float), cudaMemcpyDeviceToHost) ) ; 
  
  float xNorm, yNorm ;
  xNorm = Norm(xdata,Nfft) ;
  yNorm = Norm(ydata,Nfft) ;

  for(unsigned long i=0;i<Nfft;i++)
    xyCorr[i] = xyCorr[i] /xNorm /yNorm /(float)Nfft;

  file = fopen("xyCorr.dat","wb");
  fwrite(xyCorr, sizeof(float), Nfft, file) ;
  fclose(file) ;

  ///////////////////////////
  
  cudaFree(dev_xyCorr) ;
  cudaFreeHost(xyCorr) ;

  ///////////////////////////
  
  return 0 ;
}

