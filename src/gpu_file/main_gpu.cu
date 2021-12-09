#include <iostream>
#include <fstream>

__global__ void cuda_hello()
{
	//printf("Hello World From GPU from block %d, thread %d!\n", blockIdx.x, threadIdx.x);

	unsigned int index = blockIdx.x + threadIdx.x;

	printf("I'm %d, %d and i compute the pixel %d\n", blockIdx.x, threadIdx.x, index);

}

__global__ void vecAdd(double *a, double *b, double *c, int n,
		int blockSize, int gridSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	while (id < n)
	{
		c[id] = a[id] + b[id];
		printf("I'm %d, %d and i compute the element %d\n", blockIdx.x, threadIdx.x, id);
		id += blockSize * gridSize;
	}

}


int main(void)
{
	printf("Start the main function\n");

	double *h_a;
	double *h_b;

	double *h_c;

	double *d_a;
	double *d_b;

	double *d_c;

	size_t bytes = n * sizeof(double);

	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	for (i = 0; i < n; i++)
	{
		h_a[i] = sin(i) * sin(i);
		h_b[i] = cos(i) * cos(i);
	}

	cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	blockSize = 5;
	//gridSize = (int)ceil((float)n/blockSize);
	gridSize = 2;

	//printf("%d, %d\n", gridSize, blockSize);
	vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, blockSize, gridSize);
	//cuda_hello<<<2, 2>>>();
	cudaDeviceSynchronize();

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	for (i = 0; i < n; i++)
	{
		printf("%f ", h_c[i]);
	}



	return 0;
}
