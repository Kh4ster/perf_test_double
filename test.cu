__global__ void kernel_A(double *A, int N, int M)
{
    double d = 0.0;
    double e = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {

#pragma unroll(100)
        for (int j = 0; j < M; ++j)
        {
            d += A[idx];
            e += A[idx];
        }

        A[idx] = d + e;
    }
}

__global__ void kernel_B(double *A, int N, int M)
{
    double d = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {

#pragma unroll(100)
        for (int j = 0; j < M; ++j)
        {
            d += A[idx];
        }

        A[idx] = d;
    }
}

int main()
{

    double *A;

    int N = 80 * 2048 * 100; // 100 * maximum number of resident threads on V100
    size_t sz = N * sizeof(double);

    cudaMalloc((void **)&A, sz);

    cudaMemset(A, 1, sz);

    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int M = 1000;

    kernel_A<<<numBlocks, threadsPerBlock>>>(A, N, M);
    kernel_B<<<numBlocks, threadsPerBlock>>>(A, N, M);

    cudaDeviceSynchronize();
}
