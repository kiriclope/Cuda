#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include "librairies.h"

#include "devFunctionProtos.h"
#include "devHostConstants.h"
#include "Matrix_Utils.cu"
#include "GenConProbDistDepMat.cu"

///////////////////////////////////////////////////////////////////    

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)

void __cudaCheckLastError(const char* errorMessage, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError (msg, __FILE__, __LINE__)

void __cudaCheck(cudaError err, const char *file, const int line) {
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
      file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

void __cudaCheckLastError(const char *errorMessage, const char *file, const int line) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

///////////////////////////////////////////////////////////////////    

__host__ void nbNeurons(int N, int* &Nk) {
  
  cudaCheck(cudaMallocHost((void **)&Nk, nbpop * sizeof(int)));
  printf("Number of neurons : ") ;
  int i = 0;
  while(i<nbpop) {
    if(i==0)
      Nk[i] = N_NEURONS*popSize ;
    else
      Nk[i] = ( N_NEURONS * (100 - int( popSize * 100 ) ) / 100 ) / max( (nbpop-1), 1 ) ;
    
    printf("%d ", Nk[i]) ;
    ++i ;
  }
  printf("\n") ;
}

///////////////////////////////////////////////////////////////////    
 
__host__ void CptNeurons(int *Nk, int* &Cpt) {
  cudaCheck(cudaMallocHost((void **)&Cpt, nbpop * sizeof(int)));
  printf("Counter : ") ;
  for(int i=0;i<nbpop+1;i++) {
    Cpt[i] = 0 ;
    for(int j=0;j<i;j++) {
      Cpt[i] = Cpt[i] + Nk[j] ; 
    }
    printf("%d ", Cpt[i]) ;
  }
  printf("\n") ;
}

__global__ void initConVec(float *dev_conVec, int maxNeurons) {
  unsigned long int id = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int i;
  if(id < maxNeurons) 
    for(i = 0; i < N_NEURONS; i++) 
      dev_conVec[i + id * maxNeurons] = 0 ; 
}

__global__ void initPreFactor(float *dev_preFactor) {
  unsigned long int i;
  for(i = 0; i < 2 * N_NEURONS ; i++) 
    dev_preFactor[i] = 0 ; 
}

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
  unsigned long int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets different seed, a different sequence number, no offset */
  if(id < N_NEURONS) 
    curand_init(seed * (id + 7), id, 0, &state[id]);
}

__device__ float randkernel(curandState *state, unsigned long int kNeuron) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  /*  unsigned int id = (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;*/
  float randNumber= 0.0;
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_uniform(&localState);
    state[kNeuron] = localState;
  }
  return randNumber;
}

__global__ void kernelGenConMat(curandState *state, float *dev_conVec, int lChunck, int maxNeurons, int* nbN) {

  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  
  if(id < maxNeurons & kNeuron < N_NEURONS) 
    for(i=0; i<N_NEURONS; i++) // j is row and id is clmn 
      if( K/(float) nbN[whichPop(i)] >= randkernel(state, kNeuron)) // neuron[id] receives input from j ?
	dev_conVec[id + i * maxNeurons] = 1; 
}

__global__ void KernelGenDistDepConMat(curandState *state, float *dev_conVec, int lChunck, int maxNeurons) {
  /* GENERATE CONNECTION MATRIX WITH ANOTOMIC CONNECTIVITY PROFILE */
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long int id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons, i;
  
  if(id < maxNeurons & kNeuron < N_NEURONS)
    for(i=0; i<N_NEURONS; i++) 
      if(dev_conVec[id + i * maxNeurons] >= randkernel(state, kNeuron)) /* neuron[id] receives input from i ? */
  	dev_conVec[id + i * maxNeurons] = 1 ;
      else
  	dev_conVec[id + i * maxNeurons] = 0 ; 
}


int main(int argc, char *argv[]) {

  // int nbpop = (int) atoi(argv[1]) ; // number of neuronal populations
  // int N = (int) atoi(argv[2]) ; // total number of neurons prefactor, Ntot = N*10000.
  // double K = (double) atof(argv[3]) ;  // average number of connections

  int N = N_NEURONS ;

  int *nbN, *Cpt ;
  nbNeurons(N, nbN);
  CptNeurons(nbN, Cpt);
  
  // ///////////////////////////////////////////////////////////////////    
  
  int nChunks = 1, deviceId = 0, maxNeurons = N_NEURONS;
  float *dev_conVecPtr, *conVec = NULL, *dev_preFactor, *preFactor = NULL;
  float *fullConVec = NULL ;
  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;

  // ///////////////////////////////////////////////////////////////////    

  enum ConMat_type {
    random,distDependent
  };

  ConMat_type conMatType = random ; 
  if(IF_SPACE) {
    printf("Generating Spatial Matrix ... \n") ; 
    conMatType = distDependent ;
  }
  else
    printf("Generating Random Matrix ... \n") ; 

  ///////////////////////////////////////////////////////////////////

  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
  printf("Global Mem = %ld, ", prop.totalGlobalMem);
  maxMem = prop.totalGlobalMem;

  if(maxMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 4)) {
    while( maxMem < ( (N_NEURONS / nChunks) * N_NEURONS * 4   + N_NEURONS * 5 ) ) 
      nChunks += 1 ;
    
    if( nChunks % 2 !=0 )
      nChunks += 1 ;
  }
  
  maxNeurons = N_NEURONS / nChunks;

  if(IF_CHUNKS) {
    nChunks = NCHUNKS ;
    maxNeurons = MAXNEURONS ;
  }

  printf(" maxNeurons = %d, nChunks = %d\n", maxNeurons, nChunks);

  curandState *devStates;
  
  // printf("Declare fullConVec array ...");
  fullConVec = (float *) malloc((unsigned long long) N_NEURONS * N_NEURONS * sizeof(float));
  // printf(" Done\n");

  if(fullConVec == NULL) {
    printf("fullconvec not assigned\n"); 
    exit(-1);
  }

  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 256;  
  int BlocksPerGrid = ( N_NEURONS + ThreadsPerBlock-1 ) / ThreadsPerBlock;
  
  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }
  
  ////////////////////////////////////////////////////////////////////    

  // printf("Declare devstates array ...");
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  // printf(" Done \n") ; 

  // printf("Declare conVec array ...");
  cudaCheck(cudaMallocHost((void **)&conVec, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float)));
  // printf(" Done \n") ;

  // printf("Declare dev_conVecPtr array ...");
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float)));
  // printf(" Done \n") ;

  // printf("Declare dev_conVecPtr array ...");
  cudaCheck(cudaMalloc((void **)&dev_preFactor, 2 * N_NEURONS * sizeof(float)));
  // printf(" Done \n") ;

  // printf("Declare dev_conVecPtr array ...");
  cudaCheck(cudaMallocHost((void **)&preFactor, 2 * N_NEURONS * sizeof(float)));
  // printf(" Done \n") ;
    
  printf("Setup kernel ... \n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  cudaCheckLastError("setup_kernel failed\n");

  ///////////////////////////////////////////////////////////////////

  unsigned long long int chunckSize = ( (unsigned long long) N_NEURONS / nChunks) * N_NEURONS ;
  printf("chunckSize = %llu, ", chunckSize);
  BlocksPerGrid = (maxNeurons + ThreadsPerBlock - 1) / ThreadsPerBlock;
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);

  ///////////////////////////////////////////////////////////////////
  
  double *host_Sigma ;
  cudaCheck(cudaMallocHost((void **)&host_Sigma,  nbpop * sizeof(double))); 
  for(int j=0;j<nbpop;j++) 
    host_Sigma[j] = Sigma[j] ;
  
  if(IF_SPACE) {
    printf("Sigma ") ;
    for(int j=0;j<nbpop;j++) 
      printf("%.4f ",Sigma[j]) ;
    printf("\n") ;
  }
            
  switch(conMatType) {
    
  case random:
    
    for(unsigned long long int i = 0; i < nChunks; i++) { 
      
      printf("Generating chunk %llu ... ", i) ; fflush(stdout) ;
      
      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, maxNeurons);

      printf("\n Generating Binary Matrix ...") ;
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN); 
      printf(" Done\n") ; 

      printf("  Copy dev to Host ... ") ;
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, ( N_NEURONS/ nChunks ) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost)) ;
      printf(" Done\n") ;
      
      for(unsigned long long int j = 0; j < chunckSize ; j++) 
	fullConVec[j + chunckSize * i] = conVec[j] ; 
    }
    
    break;
      
  case distDependent:
      
    initPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_preFactor);
    
    for(unsigned long long int i = 0; i < nChunks; i++) { 
      
      printf("Generating chunk %llu ... ", i); fflush(stdout);
	
      printf("\n Generating Probabilty Matrix ...");
      KernelGenConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma);      
      printf(" Done\n") ; 
      
      printf("  Generating preFactor ...");
      KernelConProbPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ;      
      printf(" Done\n") ; 

      printf("   Copy dev to Host ...") ;
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, ( N_NEURONS/ nChunks ) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost)) ;
      printf(" Done\n") ; 

      for(unsigned long long int j = 0; j < chunckSize ; j++) {
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

    for(unsigned long long int i = 0; i < nChunks; i++) { 

      printf("Generating chunk %llu ... ", i); fflush(stdout);
      
      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, maxNeurons);

      for(unsigned long long int j = 0; j < chunckSize ; j++) 
	conVec[j] = fullConVec[j + chunckSize * i] ; 

      printf("\n Copy Host to dev ...") ;
      cudaCheck(cudaMemcpy(dev_conVecPtr, conVec, ( N_NEURONS/ nChunks ) * N_NEURONS * sizeof(float), cudaMemcpyHostToDevice)) ;
      printf(" Done\n") ; 
      
      printf("  Generating Normalized Matrix ...") ;
      KernelConProbNorm<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ;
      printf(" Done\n") ; 
            
      printf("   Generating Binary Matrix ...") ;
      KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons) ; 
      printf(" Done\n") ; 
      
      printf("    Copy dev to Host ...") ;
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, ( N_NEURONS/ nChunks ) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost)) ;
      printf(" Done\n") ; 
      
      for(unsigned long long int j = 0; j < chunckSize ; j++) {

	// if(normConVec[j]!=N_NEURONS/nbpop) {
	//   printf("\n ERRROR Chunk %llu normConVec[%llu] = %.0f \n", i, j, conVec[j] ) ;
	//   exit(-1) ;
	// }

	fullConVec[j + chunckSize * i] = conVec[j] ; 	
	conVec[j] = 0 ;
      }

    }

    break;    
    
  default:
    for(unsigned long long int i = 0; i < nChunks; i++) 
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN);
  }
  
  printf("Free devPtr ... ");

  cudaFree(dev_conVecPtr); 
  cudaFree(dev_preFactor);  

  cudaFreeHost(host_Sigma); 
  cudaFreeHost(conVec); 
  cudaFreeHost(preFactor); 

  printf("Done\n") ;

  ///////////////////////////////////////////////////////////////////    

  ///////////////////////////////////////////////////////////////////    
  // On CPU generate sparse representation
  ///////////////////////////////////////////////////////////////////    

  ///////////////////////////////////////////////////////////////////    
  
  printf("Generating sparse representation ... \n"); fflush(stdout);
  
  // printf("Declaring IdPost ... ");
  int *IdPost ; // Id of the post neurons
  IdPost = (int *) malloc((unsigned long long)N_NEURONS * (2ULL + (unsigned long long)K + N_NEURONS) * sizeof(int));
  // printf("Done\n") ;

  // printf("Declaring nbPost ... "); 
  int *nbPost = (int *) malloc( N * sizeof(*nbPost) ) ; // number of post neurons
  unsigned long int *idxPost = (unsigned long int *) malloc( N * sizeof(*idxPost)); // idx of the post neurons
  unsigned long int counter = 0; 
  // printf("Done\n") ;

  // printf("Declaring nbPreSab ... ");
  int **nbPreSab = (int **)malloc(nbpop * sizeof(int *));
  for(int i=0; i<nbpop; i++)
    nbPreSab[i] = (int *) malloc(nbpop * sizeof(int));
  // printf("Done\n") ;
    
  idxPost[0] = 0 ;
  
  ///////////////////////////////////////////////////////////////////    
  
  printf("Generating vectors nbPost & IdPost ... ");

  for(int i=0;i<nbpop;i++) 
    for(int j=0;j<nbpop;j++) 
      nbPreSab[j][i] = 0 ; 

  for(int i=0;i<nbpop;i++) 
    for(int k=Cpt[i];k<Cpt[i+1];k++) { //Presynaptic neurons
      for(int j=0;j<nbpop;j++) 
  	for(int l=Cpt[j];l<Cpt[j+1];l++) //Postsynaptic neurons
  	  if(fullConVec[k + N_NEURONS * l]) { // k-->l column to row
	    IdPost[counter] = l ;
	    nbPost[k]++ ;
	    nbPreSab[j][i]++ ;
  	    counter+=1 ;
  	  }   
      // printf("PresId %d, nPost %d \r",k,nbPost[k]);
    }
  printf("Done\n") ;

  ///////////////////////////////////////////////////////////////////    
  // Average number of Presynaptic neurons
  ///////////////////////////////////////////////////////////////////    

  char *path = '\0';
  CreatePath(path,N) ;
  
  CheckPres(path,nbN,nbPreSab) ;
  free(nbPreSab);

  ///////////////////////////////////////////////////////////////////    
  // Writing to File
  ///////////////////////////////////////////////////////////////////

  WritetoFile(path,N,IdPost,nbPost,idxPost) ;

  ///////////////////////////////////////////////////////////////////    
  // Writing Complete Matrix
  ///////////////////////////////////////////////////////////////////

  if(IF_MATRIX)
    WriteMatrix(path,N,IdPost,nbPost,idxPost) ;

  printf("Free Sparse Vectors ... ") ;

  cudaFreeHost(nbN); 
  cudaFreeHost(Cpt); 

  free(fullConVec);
  free(IdPost);
  free(idxPost);
  free(nbPost);

  printf("Done\n") ;

  return 0 ;
  
}