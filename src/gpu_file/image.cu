#include "image.hh"

#include <iostream>
#include <stdlib.h>


/* GLOBALfunction */

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


/* Function for Image class */


Image::Image(const char* path, int pool_size_arg)
{

    img_array = stbi_load(path, &width, &height, &nb_chan, 0);

    if (img_array == NULL)
    {
        std::cout << "Error : can't open the image: " << path << "\n";
    }

    pool_size = pool_size_arg;
    nb_patch_x =  width / pool_size;
    nb_patch_y =  height / pool_size;

    img_gray_array = new unsigned char[width * height];
    img_sobel_x_array = new unsigned char[width * height];
    img_sobel_y_array = new unsigned char[width * height];
    img_sobel_patch_x_array = new unsigned char[patch_size_x * patch_size_y];
    img_sobel_patch_y_array = new unsigned char[patch_size_x * patch_size_y];
}

void Image::print_image()
{

    printf("The image have a size of %d x %d x %d\n", width, height, nb_chan);
    printf("The pool size is %d\n", pool_size);
    printf("And so the image have %d patch on x and %d patch en y\n", patch_size_x, patch_size_y);

}

Image::~Image()
{
    if (img_array)
    {
        stbi_image_free(img_array);
    }

    free(img_gray_array);
    free(img_sobel_x_array);
    free(img_sobel_y_array);
}


void Image::save_gray_img()
{
    stbi_write_jpg("../../img/codebar_gray.jpg", width, height, 1,
            img_gray_array, 100);
}


void Image::save_sobel_img()
{

    stbi_write_jpg("../../img/codebar_sobel_x.jpg", width, height, 1,
            img_sobel_x_array, 100);
    stbi_write_jpg("../../img/codebar_sobel_y.jpg", width, height, 1,
            img_sobel_y_array, 100);
}


void Image::save_sobel_img()
{

    stbi_write_jpg("../../img/codebar_patch_x.jpg", width, height, 1,
        img_patch_x_array, 100);

}

void Image::create_gray_array()
{

    unsigned char *d_gray_img;
    unsigned char *d_img;

    size_t gray_img_size = width * height * sizeof(unsigned char);
    size_t img_size = width * height * 3 * sizeof(unsigned char);

    cudaMalloc(&d_gray_img, gray_img_size);
    cudaMalloc(&d_img, img_size);


    cudaMemcpy( d_gray_img, img_gray_array, gray_img_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_img, img_array, img_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_gray<<<gridSize, blockSize>>>(d_gray_img, d_img, width, height,
                                            blockSize, gridSize);
    cudaDeviceSynchronize();

    cudaMemcpy(img_gray_array, d_gray_img, gray_img_size, cudaMemcpyDeviceToHost);

}

void Image::create_sobel_array()
{
    unsigned char *d_sobel_x;
    unsigned char *d_sobel_y;
    unsigned char *d_gray_img;

    size_t img_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_sobel_x, img_size);
    cudaMalloc(&d_sobel_y, img_size);
    cudaMalloc(&d_gray_img, img_size);


    cudaMemcpy( d_sobel_x, img_sobel_x_array, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_sobel_y, img_sobel_y_array, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_gray_img, img_gray_array, img_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_sobel<<<gridSize, blockSize>>>(d_sobel_x, d_sobel_y, d_gray_img,
                                        width, height, blockSize, gridSize);
    cudaDeviceSynchronize();

    cudaMemcpy(img_sobel_x_array, d_sobel_x, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_sobel_y_array, d_sobel_y, img_size, cudaMemcpyDeviceToHost);

}

void Image::create_patch_array()
{
    
}

int Image::get_size()
{
    return width * height;
}


int main(void)
{
    Image image("../../img/codebar.jpg", 8);
    image.print_image();
    image.create_gray_array();
    image.save_gray_img();
    image.create_sobel_array();
    image.save_sobel_img();

    return 0;
}
