/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#define _DEFAULT_BLOCK_SIZE 32
#define _ATOMICADD
#define _ACCELERATE_KERNEL
#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

__global__ void _vecadd(const double * in, double * out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N) return;
    out[i] += in[i];
}

/* ---------------------------------------------------------------------- */

void my_gpu_vecadd(const double * in, double * out, int N)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  _vecadd<<<grid_size, block_size>>>(in, out, N);
}


/* ---------------------------------------------------------------------- */

__global__ void _vecadd_batch(const double * in, double * out, int N, int num_batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N) return;

    double val = 0.0;
    for(int j=0; j<num_batches; ++j) val += in[j*N + i];
    
    out[i] += val;
}

/* ---------------------------------------------------------------------- */

void my_gpu_vecadd_batch(const double * in, double * out, int N, int num_batches)
{
  dim3 block_size(_DEFAULT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(N,block_size.x));
  
  _vecadd_batch<<<grid_size, block_size>>>(in, out, N, num_batches);
}

/* ---------------------------------------------------------------------- */

#endif
