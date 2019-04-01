__device__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}



__device__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}



__global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale) {

  const int numThreads = blockDim.x * gridDim.x;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = index; i < size; i += numThreads)    
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);     
}

__global__ void real2complex(float *f, cufftComplex *fc, int N) {
  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int index = j*N+i;
  
  if(i<N && j<N) {
    fc[index].x = f[index];
    fc[index].y = 0.0f;
  }
}
__global__ void complex2real(cufftComplex *fc, float *f, int N) {
  int i = threadIdx.x + blockIdx.x*BSZ;
  int j = threadIdx.y + blockIdx.y*BSZ;
  int index = j*N+i;
  if(i<N && j<N) {
    f[index] = fc[index].x/((float)N*(float)N);
    //divide by number of elements to recover value
  }
}
