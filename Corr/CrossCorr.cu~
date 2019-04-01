///////////////////////////////////////////////

__global__ void initReal(cufftReal *dev_vec, unsigned long VecSize) {
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;

  if(id < VecSize) 
    dev_vec[id] = 0 ; 

}

///////////////////////////////////////////////

__global__ void initComplex(cufftComplex *dev_vec, unsigned long VecSize) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;

  if(id < VecSize) {
    dev_vec[id].x = 0 ; 
    dev_vec[id].y = 0 ; 
  }

}

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
  
  unsigned long id =  (unsigned long) threadIdx.x + blockIdx.x * blockDim.x;
  cufftComplex c;  
  
  if( id < dataSize ) {
    c = (cufftComplex) cuConjf( (cuComplex) dev_ydata[id] ) ;
    dev_zdata[id] = (cufftComplex) ComplexMul( dev_xdata[id], c ) ;
    // cuPrintf("id %d zdata %f +i%f\r", id, dev_zdata[id].x, dev_zdata[id].y) ;
  }  
}

///////////////////////////////////////////////

__host__ void AllocateDevicePtr(unsigned long Nfft, cufftReal *dev_xdata, cufftReal *dev_ydata, cufftReal *dev_xyCorr, cufftComplex *dev_iXdata, cufftComplex *dev_iYdata, cufftComplex *dev_iZdata) {
  
  cudaCheck( cudaMalloc((void**)&dev_xdata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_ydata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_xyCorr, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  
  cudaCheck( cudaMalloc((void**)&dev_iXdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iYdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iZdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  
}

///////////////////////////////////////////////

__host__ void CrossCorr(unsigned long Nfft, float *xdata, float *ydata, float *xyCorr, cufftHandle fftplan, cufftHandle ifftplan, cufftReal *dev_xdata, cufftReal *dev_ydata, cufftReal *dev_xyCorr, cufftComplex *dev_iXdata, cufftComplex *dev_iYdata, cufftComplex *dev_iZdata) {
  
  ///////////////////////////////////////
  // Kernel Setup 
  ///////////////////////////////////////

  int ThreadsPerBlock = N_THREADS ;
  int BlocksPerGrid = ( Nfft + ThreadsPerBlock-1 ) / ThreadsPerBlock;

  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }
  
  ///////////////////////////
  // cufft utils
  ///////////////////////////

  // printf("... Done\nAllocate device pointers ") ;

  initReal<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_xdata, Nfft) ;
  cudaCheckLastError("dev_xdata initialisation failed\n");
  initReal<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_ydata, Nfft) ;
  cudaCheckLastError("dev_ydata initialisation failed\n");
  initReal<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_xyCorr, Nfft) ;
  cudaCheckLastError("dev_xyCorr initialisation failed\n");

  initComplex<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_iXdata, (Nfft/2+1)) ;
  cudaCheckLastError("dev_iXdata initialisation failed\n");
  initComplex<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_iYdata, (Nfft/2+1)) ;
  cudaCheckLastError("dev_iYdata initialisation failed\n");
  initComplex<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_iZdata, (Nfft/2+1)) ;
  cudaCheckLastError("dev_iZdata initialisation failed\n");
    
  // printf("... Done\nCopy Host to Dev") ;
  
  cudaCheck( cudaMemcpy(dev_xdata, xdata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH, cudaMemcpyHostToDevice) ) ; 
  cudaCheck( cudaMemcpy(dev_ydata, ydata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH, cudaMemcpyHostToDevice) ) ; 
  
  ///////////////////////////

  // printf("... Done\nFirst FFT ") ;
  /* Use the CUFFT plan to transform the signal in place. */
  if (cufftExecR2C(fftplan, dev_xdata, dev_iXdata) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecR2C (1) Forward failed");
    exit(-1) ;
  }

  // printf("... Done\n Second FFT ") ;
  if (cufftExecR2C(fftplan, dev_ydata, dev_iYdata) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecR2C (2) Forward failed\n");
    exit(-1) ;
  }
  
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  ///////////////////////////////////////
  // Multiplication
  ///////////////////////////////////////

  // printf("... Done\n  Multiply FFT ") ;
  
  cudaMultiply<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_iXdata, dev_iYdata, dev_iZdata, (unsigned long) Nfft/2+1 ) ; 
  cudaCheckLastError("Multiplication failed\n");
  cudaPrintfDisplay(stdout, true);
  
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  ///////////////////////////

  // printf("... Done\n   Invert FFT ") ;
  
  if (cufftExecC2R(ifftplan, dev_iZdata, dev_xyCorr) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2R Forward failed\n");
    exit(-1) ;
  }

  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    exit(-1) ;
  }

  ///////////////////////////
  
  // printf("... Done\nFree Memory \n") ;
  
  ///////////////////////////
  
  cudaCheck( cudaMemcpy(xyCorr, dev_xyCorr, (unsigned long) Nfft * sizeof(float), cudaMemcpyDeviceToHost) ) ; 
  
  float xNorm, yNorm ;
  xNorm = Norm(xdata,Nfft) ;
  yNorm = Norm(ydata,Nfft) ;

  for(unsigned long i=0;i<Nfft;i++)
    // xyCorr[i] = xyCorr[i] /(float) Nfft;
    xyCorr[i] = xyCorr[i] /xNorm /yNorm /(float) Nfft;

  ///////////////////////////
  
}

