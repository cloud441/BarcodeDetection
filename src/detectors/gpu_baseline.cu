#include "gpu_baseline.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

GPUBaseline::GPUBaseline() {}

void GPUBaseline::load_img(std::string path, int scale)
{
    img_array = stbi_load(path.c_str(), &width, &height, &nb_chan, 0);

    if (img_array == NULL)
    {
        std::cout << "Error : can't open the image: " << path << "\n";
    }

    img_gray_array = new unsigned char[width * height];
    img_sobel_x_array = new unsigned char[width * height];
    img_sobel_y_array = new unsigned char[width * height];
}

__global__ void compute_gray(unsigned char *d_gray_array,
                             unsigned char *d_array, int width, int weight,
                             int blockSize, int gridSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = width * weight;

    while (id < img_size)
    {
        d_gray_array[id] = d_array[id * 3] * 0.2989 +
                           d_array[id * 3 + 1] * 0.5870 + d_array[id * 3 + 2] * 0.1140;

        id += blockSize * gridSize;
    }
}

__global__ void compute_sobel(unsigned char *d_sobel_x,
                              unsigned char *d_sobel_y, unsigned char *d_gray_array, int width,
                              int weight, int blockSize, int gridSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = width * weight;

    while (id < img_size)
    {
        // We will compute 9 pixel withe the next shape

        // 1 2 3
        // 4 5 6
        // 7 8 9

        // So we will compte only the pixel that are not in the border

        if (id <= width)
        {
            d_sobel_x[id] = 0;
            d_sobel_y[id] = 0;
        }
        else if (id % width == 0)
        {
            d_sobel_x[id] = 0;
            d_sobel_y[id] = 0;
        }
        else if (id % width == (width - 1))
        {
            d_sobel_x[id] = 0;
            d_sobel_y[id] = 0;
        }
        else if (id >= (img_size - width))
        {
            d_sobel_x[id] = 0;
            d_sobel_y[id] = 0;
        }
        else
        {
            int sum_x = 0;
            int sum_y = 0;

            sum_x -= d_gray_array[id - 1 - width];
            sum_x += d_gray_array[id + 1 - width];
            sum_x -= 2 * d_gray_array[id - 1];
            sum_x += 2 * d_gray_array[id + 1];
            sum_x -= d_gray_array[id - 1 + width];
            sum_x += d_gray_array[id + 1 + width];

            sum_y -= d_gray_array[id - 1 - width];
            sum_y -= 2 * d_gray_array[id - width];
            sum_y -= d_gray_array[id + 1 - width];
            sum_y += d_gray_array[id - 1 + width];
            sum_y += 2 * d_gray_array[id + width];
            sum_y += d_gray_array[id + 1 + width];

            sum_x = abs(sum_x);
            sum_y = abs(sum_y);

            d_sobel_x[id] = sum_x;
            d_sobel_y[id] = sum_y;
        }

        id += blockSize * gridSize;
    }
}

void GPUBaseline::create_gray_array()
{
    /*
    // CPU Version
    for (int i = 0; i < height * width; ++i)
    {
        img_gray_array[i] = (uint8_t)((img_array[i * 3] +
                    img_array[i * 3 + 1] + img_array[i * 3 + 2])/3.0);
    }
    */

    unsigned char *d_gray_img;
    unsigned char *d_img;

    size_t gray_img_size = width * height * sizeof(unsigned char);
    size_t img_size = width * height * 3 * sizeof(unsigned char);

    cudaMalloc(&d_gray_img, gray_img_size);
    cudaMalloc(&d_img, img_size);

    cudaMemcpy(d_gray_img, img_gray_array, gray_img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img, img_array, img_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_gray<<<gridSize, blockSize>>>(d_gray_img, d_img, width, height,
                                          blockSize, gridSize);
    cudaDeviceSynchronize();

    cudaMemcpy(img_gray_array, d_gray_img, gray_img_size, cudaMemcpyDeviceToHost);
}

void GPUBaseline::compute_derivatives()
{
    unsigned char *d_sobel_x;
    unsigned char *d_sobel_y;
    unsigned char *d_gray_img;

    size_t img_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_sobel_x, img_size);
    cudaMalloc(&d_sobel_y, img_size);
    cudaMalloc(&d_gray_img, img_size);

    cudaMemcpy(d_sobel_x, img_sobel_x_array, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_y, img_sobel_y_array, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gray_img, img_gray_array, img_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_sobel<<<gridSize, blockSize>>>(d_sobel_x, d_sobel_y, d_gray_img,
                                           width, height, blockSize, gridSize);
    cudaDeviceSynchronize();

    cudaMemcpy(img_sobel_x_array, d_sobel_x, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_sobel_y_array, d_sobel_y, img_size, cudaMemcpyDeviceToHost);
}

void GPUBaseline::save_gray_img()
{
    stbi_write_jpg("../../img/codebar_gray.jpg", width, height, 1,
                   img_gray_array, 100);
}

void GPUBaseline::save_sobel_img()
{

    for (int i = 0; i < width * height; i++)
    {
        printf("x : %d y : %d\n", img_sobel_x_array[i], img_sobel_y_array[i]);
    }

    stbi_write_jpg("../../img/codebar_sobel_x.jpg", width, height, 1,
                   img_sobel_x_array, 100);
    stbi_write_jpg("../../img/codebar_sobel_y.jpg", width, height, 1,
                   img_sobel_y_array, 100);
}
