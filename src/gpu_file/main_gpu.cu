#include "image.hh"

#include <iostream>
#include <fstream>

int main(void)
{
    Image Image("codebar.jpg");

    return 0;
}

int main_temp(void)
{
/*
    printf("Start the main function\n");

    int n = 20;

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

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, blockSize, gridSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++)
    {
        printf("%f ", h_c[i]);
    }

*/

    return 0;
}
