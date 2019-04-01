#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include "librairies.h"
#include "cuPrintf.cu"
#include "devFunctionProtos.h"
#include "devHostConstants.h"
#include "CudaFunc.cu"
#include "Matrix_Utils.cu"
#include "GenConProbDistDepMat.cu"

///////////////////////////////////////////////////////////////////  

int main(int argc, char *argv[]) {
  
  char* AtoB ;
  AtoB =  (char *) malloc( strlen("EE") ) ;

  if(IF_LARGE)
    AtoB = argv[1] ;
  
  unsigned long int *nbN, *Cpt ;
  nbNeurons(nbN) ;
  CptNeurons(nbN, Cpt) ;
  
  // ///////////////////////////////////////////////////////////////////    
  
  unsigned long nChunks = 1, deviceId = 0 ;
  unsigned long maxNeurons = NX_NEURONS ; 

  ///////////////////////////////////////////////////////////////////
  
  cudaDeviceProp prop;
  unsigned long long maxMem = 12079136768 ; 

  cudaCheck(cudaGetDeviceProperties(&prop, deviceId)) ;
  printf("Global Mem = %ld, ", prop.totalGlobalMem) ; 
  maxMem = prop.totalGlobalMem;

  if( maxMem < (unsigned long long) (NX_NEURONS * NY_NEURONS * 4 + NX_NEURONS * 4) ) { 
    while( maxMem < (unsigned long) ( (NX_NEURONS / nChunks) * NY_NEURONS * 4   + NX_NEURONS * 5 ) ) 
      nChunks += 1 ;     
    if( nChunks % 2 !=0 ) 
      nChunks += 1 ; 
  }

  maxNeurons = (unsigned long) NX_NEURONS / nChunks ; //divide the presynaptic into nChunks 

  if(IF_CHUNKS) { 
    nChunks = NCHUNKS ; 
    maxNeurons = MAXNEURONS ; 
  }

  printf(" maxNeurons = %lu, nChunks = %lu\n", maxNeurons, nChunks); 

  /////////////////////////////////////////////////////////////////// 

  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = N_THREADS ; 
  int BlocksPerGrid = ( NX_NEURONS + ThreadsPerBlock-1 ) / ThreadsPerBlock ; 
  
  if(BlocksPerGrid > 65536) { 
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536"); 
    exit(-1); 
  } 
  
  curandState *devStates; 
  cudaCheck(cudaMalloc((void **)&devStates, NX_NEURONS * sizeof(curandState))); 
  
  printf("Setup kernel ... \n"); 
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  cudaCheckLastError("setup_kernel failed\n"); 

  /////////////////////////////////////////////////////////////////// 

  unsigned long long chunckSize = (unsigned long long) ( NX_NEURONS / nChunks * NY_NEURONS ) ; 
  printf("chunckSize = %llu, ", chunckSize);

  BlocksPerGrid = (maxNeurons + ThreadsPerBlock - 1) / ThreadsPerBlock ; 
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);

  // ///////////////////////////////////////////////////////////////////    

  float *dev_conVecPtr, *dev_preFactor ; //*preFactor = NULL ; 
  float *fullConVec = NULL, *conVec = NULL ; 
  unsigned long *IdPost ; 
  int *nbPost ; 

  ///////////////////////////////////////////////////////////////////
  
  fullConVec = (float *) malloc((unsigned long long) NX_NEURONS * NY_NEURONS * sizeof(float)) ;

  //NX_NEURONS presynaptic and NY_NEURONS postsynaptic 
  IdPost = (unsigned long *) malloc( (unsigned long long) NX_NEURONS * ( K + sqrt(K) ) * sizeof( unsigned long ) ) ; // idx of the postsynaptic neurons from 1 to NY_NEURONS on average K * NX_NEURONS +- sqrt(K) 
  nbPost = (int *) malloc( (unsigned long long) NX_NEURONS * sizeof(int)) ; // number of postsynaptic neurons of the NX_NEURONS

  int **nbPreSab = (int **)malloc(nbpop * sizeof(int *) ) ; 
  for(int i=0; i<nbpop; i++) 
    nbPreSab[i] = (int *) malloc(nbpop * sizeof(int) ) ; 
  for(int i=0; i<nbpop; i++) 
    for(int j=0;j<nbpop;j++) 
      nbPreSab[i][j] = 0 ;
  
  ////////////////////////////////////////////////////////////////////    

  cudaCheck(cudaMallocHost((void **)&conVec, (unsigned long long) chunckSize * sizeof(float))) ;  
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float))) ; 
  cudaCheck(cudaMalloc((void **)&dev_preFactor, (unsigned long long) nbpop * NX_NEURONS * sizeof(float))) ; 
  
  // cudaCheck(cudaMallocHost((void **)&preFactor, 2 * N_NEURONS * sizeof(float))); 

  ///////////////////////////////////////////////////////////////////

  enum ConMat_type {
    random, distDependent
  };

  ConMat_type conMatType = random ; 
  if(IF_SPACE) {
    printf("Generating Spatial Matrix ... \n") ; 
    conMatType = distDependent ;
  }
  else 
    if(IF_RING) 
      printf("Generating Ring ... \n") ; 
    else 
      printf("Generating Random Matrix ... \n") ; 
  
  if(IF_SPEC) 
    printf("with specific connections ... \n") ; 

  if(IF_AUTA) 
    printf("with autapses ... \n") ; 
  
  ///////////////////////////////////////////////////////////////////
  
  double *host_Sigma ;
  cudaCheck(cudaMallocHost((void **)&host_Sigma,  nbpop * sizeof(double))) ; 
  for(int j=0;j<nbpop;j++) 
    host_Sigma[j] = Sigma[j] ; 
  if(IF_RING & IF_SPEC) 
    for(int j=0;j<nbpop;j++) 
      host_Sigma[j] = host_Sigma[j]/sqrt(K) ; 

  double *host_Dij ; 
  cudaCheck(cudaMallocHost((void **)&host_Dij,  nbpop * sizeof(double))) ; 
  if(IF_RING & IF_SPEC) 
    for(int j=0;j<nbpop*nbpop;j++) 
      host_Dij[j] = Dij[j] ; 
    

  if(IF_SPACE || IF_RING) { 
    printf("Sigma ") ; 
    for(int j=0;j<nbpop;j++)  
      printf("%.4f ",Sigma[j]) ; 
    printf("\n") ; 
  } 

  cudaPrintfInit(); 

  switch(conMatType) {
    
  case random: 
    
    for(unsigned long i = 0; i < nChunks; i++) { 

      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons);
      
      printf("Generating chunk %lu ... \n", i) ; fflush(stdout) ;
      
      printf(" Generating Binary Matrix ...\n") ;
      if(IF_RING) {
	KernelGenConRing<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN, Cpt, host_Sigma, host_Dij) ; 
	KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons) ; 
	cudaPrintfDisplay(stdout, true);
      }
      else {
	if(IF_AUTA) 
	  kernelGenConAuta<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN) ; 

	kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN) ; 
	cudaPrintfDisplay(stdout, true);
	
      }
      
      printf("  Copy dev to Host ... \n") ; 
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float), cudaMemcpyDeviceToHost));
      
      for(unsigned long j = 0; j < chunckSize ; j++) {
	fullConVec[j + chunckSize * i] = (float) conVec[j] ; 
	// printf("# %llu Con %f fullConVec %f \n", j + chunckSize * i, conVec[j], fullConVec[j + chunckSize * i]) ;
	conVec[j] = 0 ;
      }
    }

    cudaPrintfEnd();

    break;
    
  case distDependent:
    initPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_preFactor);
    
    for(unsigned long i = 0; i < nChunks; i++) { 

      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons) ;

      printf("Generating chunk %lu ... \n", i); fflush(stdout);
	
      printf(" Generating Probabilty Matrix ...\n");
      if(DIMENSION==1)
	KernelGenConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma) ; 
      else
	KernelGenConProbMat2D<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma) ; 
      
      printf("  Generating preFactor ...\n");
      KernelConProbPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ;      

      printf("   Copy dev to Host ...\n") ;
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float), cudaMemcpyDeviceToHost)) ;

      for(unsigned long j = 0; j < chunckSize ; j++) {
	fullConVec[j + chunckSize * i] = conVec[j] ; 
	
	// if(conVec[j]!=1) {
	//   printf("\n ERRROR Chunk %llu conVec[%llu] = %.3f \n", i, j, conVec[j] ) ;
	//   exit(-1) ;
	// }
	
	conVec[j] = 0 ;
      }      

    }
    
    // printf("Copy preFactor to Host ...") ; 
    // cudaCheck(cudaMemcpy(preFactor, dev_preFactor, 2 * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost) ) ; 
    // printf(" Done\n ") ; 
    
    // printf(" Check preFactor ...") ; 
    // for(int j=0;j<2*N_NEURONS;j++)
    //   if(preFactor[j]!=N_NEURONS/nbpop) {
    // 	printf("ERROR clm %d prefactor %.3f \n", j, preFactor[j]) ;
    // 	exit(-1) ;
    //   }
    // printf(" %.0f ", preFactor[0]) ;
    // printf(" Done\n") ; 

    for(unsigned long i = 0; i < nChunks; i++) { 

      printf("Generating chunk %lu ... \n", i); fflush(stdout);
      
      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons) ;

      for(unsigned long j = 0; j < chunckSize ; j++) 
	conVec[j] = fullConVec[j + chunckSize * i] ; 

      printf(" Copy Host to dev ...\n") ;
      cudaCheck(cudaMemcpy(dev_conVecPtr, conVec, (unsigned long long) chunckSize * sizeof(float), cudaMemcpyHostToDevice)) ;
      
      printf("  Generating Normalized Matrix ...\n") ;
      KernelConProbNorm<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ; 
            
      printf("   Generating Binary Matrix ...\n") ; 
      KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons) ; 
            
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float), cudaMemcpyDeviceToHost)) ; 
      
      for(unsigned long j = 0; j < chunckSize ; j++) { 

	// if(normConVec[j]!=N_NEURONS/nbpop) { 
	//   printf("\n ERRROR Chunk %llu normConVec[%llu] = %.0f \n", i, j, conVec[j] ) ; 
	//   exit(-1) ; 
	// }

	fullConVec[j + chunckSize * i] = conVec[j] ; 
	conVec[j] = 0 ; 
      }
      
    }
    
    break ; 
    
  default: 
    for(unsigned long i = 0; i < nChunks; i++) 
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN);
  }
  
  printf("Free devPtr ... "); 

  cudaFree(dev_conVecPtr); 
  cudaFree(dev_preFactor); 
  
  cudaFreeHost(host_Sigma); 
  cudaFreeHost(conVec); 

  printf("Done\n") ; 
  
  /////////////////////////////////////////////////////////////////// 

  // ///////////////////////////////////////////////////////////////////    
  // // On CPU 
  // ///////////////////////////////////////////////////////////////////    

  ////////////////////////////////////////////////////////////////////    
  
  unsigned long *idxPost = (unsigned long *) malloc( (unsigned long long) NX_NEURONS * sizeof(unsigned long) ) ; // idx of the post neurons 
  idxPost[0] = 0 ; 
  
  char *path = '\0';
  CreatePath(path) ; 

  if(IF_SPARSEVEC) {
    printf("Generating vectors nbPost & IdPost ... "); 
    
    unsigned long counter = 0 ; 

    if(NX_NEURONS==NY_NEURONS) 
      for(int i=0;i<nbpop;i++) 
	for(unsigned long k=Cpt[i];k<Cpt[i+1];k++) { //Presynaptic neurons
	  for(int j=0;j<nbpop;j++) 
	    for(unsigned long l=Cpt[j];l<Cpt[j+1];l++) //Postsynaptic neurons
	      if(fullConVec[k + NY_NEURONS * l]) { // k-->l column to row 
		IdPost[counter] = l ;
		nbPost[k]++ ;
		nbPreSab[j][i]++ ;
		counter+=1 ;
	      }   
	  // printf("PresId %d, nPost %d \r",k,nbPost[k]); 
	}
    else {
      printf("NX %llu NY %llu... ", NX_NEURONS, NY_NEURONS); 
      nbN[0] = NX_NEURONS ;
      for(unsigned long k=0;k<NX_NEURONS;k++) { //Presynaptic neurons
	for(unsigned long l=0;l<NY_NEURONS;l++) //Postsynaptic neurons
	  if(fullConVec[k + NX_NEURONS * l]) { // k-->l column to row 
	    IdPost[counter] = l ;
	    nbPost[k]++ ;
	    nbPreSab[0][0]++ ; 
	    counter+=1 ;
	  }
      }
      // printf("PresId %d, nPost %d \r",k,nbPost[k]);      
    }
      
    // Average number of Presynaptic neurons      
    CheckPres(path,nbN,nbPreSab) ; 
    free(nbPreSab);

    ///////////////////////////////////////////////////////////////////    
    // Writing to File
    ///////////////////////////////////////////////////////////////////

    if(IF_LARGE)
      WritetoFileLarge(path,IdPost,nbPost,idxPost,AtoB) ;
    else
      WritetoFile(path,IdPost,nbPost,idxPost) ;
  }

  free(IdPost);
  free(idxPost);
  free(nbPost);

  ///////////////////////////////////////////////////////////////////    
  // Writing Complete Matrix
  ///////////////////////////////////////////////////////////////////

  if(IF_MATRIX) {
    // CheckSparseVec(path) ;
    WriteMatrix(path,fullConVec) ;
  }

  printf("Free Host ptr ... ") ;

  cudaFreeHost(nbN); 
  cudaFreeHost(Cpt); 

  //free(fullConVec);

  printf("Done\n") ;

  return 0 ;
  
}