#include "image.hh"

#include <iostream>
#include <stdlib.h>
#include <string>

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


__global__ void compute_patch(unsigned char *d_patch_x, unsigned char *d_patch_y,
        unsigned char *d_sobel_x_array, unsigned char *d_sobel_y_array,
        int pool_size, int width, int height, int nb_patch_x, int nb_patch_y,
        int blockSize, int gridSize)
{

    int nb_patch = nb_patch_x * nb_patch_y;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    while (id < nb_patch)
    {

        //printf("%d,%d,%d start to compute\n", blockIdx.x, threadIdx.x, id);

        int id_patch_x = id % nb_patch_x;
        int id_patch_y = id / nb_patch_x;

        int id_pixel = pool_size * id_patch_y * width;
        id_pixel += id_patch_x * pool_size;

        /*
        printf("%d,%d,%d start to compute the patch %d,%d, and the pixel is %d\n",
                blockIdx.x, threadIdx.x, id, id_patch_x, id_patch_y, id_pixel);
        */

        int nb_elem = pool_size * pool_size;
        int sum_x = 0;
        int sum_y = 0;

        for (int i = 0; i < pool_size; i++)
        {
            for (int j = 0; j < pool_size; j++)
            {
                /*
                if (blockIdx.x == 1 && threadIdx.x == 1 && id == 6)
                {
                    printf("%d,%d,%d compute the pixel  %d for the pixel %d\n",
                            blockIdx.x, threadIdx.x, id,id_pixel + i * width + j, id_pixel);
                }
                */

                sum_x += d_sobel_x_array[id_pixel + i * width + j];
                sum_y += d_sobel_y_array[id_pixel + i * width + j];
            }
        }

        sum_x = sum_x / nb_elem;
        sum_y = sum_y / nb_elem;

        /*
        printf("%d,%d,%d the mean is : %d, %d\n",
                blockIdx.x, threadIdx.x, id, sum_x, sum_y);
        */



        if (sum_x < 0)
        {
            sum_x = 0;
        }
        else if (sum_x > 255)
        {
            sum_x = 255;
        }

        if (sum_y < 0)
        {
            sum_y = 0;
        }
        else if (sum_y > 255)
        {
            sum_y = 255;
        }

        d_patch_x[id_patch_y * nb_patch_x + id_patch_x] = sum_x;
        d_patch_y[id_patch_y * nb_patch_x + id_patch_x] = sum_y;

        id += blockSize * gridSize;

    }

}


__global__ void compute_response(unsigned char *d_response,
        unsigned char *d_patch_x, unsigned char *d_patch_y,
        int nb_patch_x, int nb_patch_y, int blockSize, int gridSize)
{

    int nb_patch = nb_patch_x * nb_patch_y;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    while (id < nb_patch)
    {

        int value = d_patch_x[id] - d_patch_y[id];

        if (value < 0)
        {
            value = 0;
        }
        else if (value > 255)
        {
            value = 255;
        }

        d_response[id] = value;

        id += blockSize * gridSize;

    }

}



__global__ void compute_dilatation(unsigned char *d_response,
        unsigned char *d_response_clean_1,
        int width, int weight, int blockSize, int gridSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = width * weight;

    while (id < img_size)
    {

        if (id <= 2 * width)
        {
            d_response_clean_1[id] = 0;
        }
        else if (id % width <= 1)
        {
            d_response_clean_1[id] = 0;
        }
        else if (id % width >= (width - 2))
        {
            d_response_clean_1[id] = 0;
        }
        else if (id >= (img_size - 2 * width))
        {
            d_response_clean_1[id] = 0;
        }
        else
        {
            int max_value = d_response[id];

            for (int i = -2; i <= 2; i++)
            {
                for (int j = -2; i <= 2; j++)
                {
                    int value = d_response[id + i * width + j];

                    if (value > max_value)
                    {
                        max_value = value;
                    }
                }
            }

            d_response_clean_1[id] = max_value;

        }

        id += blockSize * gridSize;
    }
}


__global__ void compute_erosion(unsigned char *d_response_clean_1,
        unsigned char *d_response_clean_2,
        int width, int weight, int blockSize, int gridSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = width * weight;

    while (id < img_size)
    {

        if (id <= 2 * width)
        {
            d_response_clean_2[id] = 0;
        }
        else if (id % width <= 1)
        {
            d_response_clean_2[id] = 0;
        }
        else if (id % width >= (width - 2))
        {
            d_response_clean_2[id] = 0;
        }
        else if (id >= (img_size - 2 * width))
        {
            d_response_clean_2[id] = 0;
        }
        else
        {
            int min_value = d_response_clean_1[id];

            for (int i = -2; i <= 2; i++)
            {
                for (int j = -2; i <= 2; j++)
                {
                    int value = d_response_clean_1[id + i * width + j];

                    if (value < min_value)
                    {
                        min_value = value;
                    }
                }
            }

            d_response_clean_2[id] = min_value;

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
    img_sobel_patch_x_array = new unsigned char[nb_patch_x * nb_patch_y];
    img_sobel_patch_y_array = new unsigned char[nb_patch_x * nb_patch_y];
    img_response_array = new unsigned char[nb_patch_x * nb_patch_y];
    img_response_clean_1_array = new unsigned char[nb_patch_x * nb_patch_y];
    img_response_clean_2_array = new unsigned char[nb_patch_x * nb_patch_y];
}


void Image::print_image()
{

    printf("The image have a size of %d x %d x %d\n", width, height, nb_chan);
    printf("The pool size is %d\n", pool_size);
    printf("And so the image have %d patch on x and %d patch en y\n", nb_patch_x, nb_patch_y);

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
    free(img_sobel_patch_x_array);
    free(img_sobel_patch_y_array);
    free(img_response_array);
    free(img_response_clean_1_array);
    free(img_response_clean_2_array);
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


void Image::save_patch_img()
{

    stbi_write_jpg("../../img/codebar_patch_x.jpg", nb_patch_x, nb_patch_y, 1,
        img_sobel_patch_x_array, 100);
    stbi_write_jpg("../../img/codebar_patch_y.jpg", nb_patch_x, nb_patch_y, 1,
        img_sobel_patch_y_array, 100);

}


void Image::save_response_img()
{
    stbi_write_jpg("../../img/codebar_response.jpg", nb_patch_x, nb_patch_y, 1,
        img_response_array, 100);
}


void Image::save_response_clean_img()
{
    stbi_write_jpg("../../img/codebar_response_clean.jpg", nb_patch_x, nb_patch_y, 1,
        img_response_clean_2_array, 100);
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
    unsigned char *d_patch_x;
    unsigned char *d_patch_y;
    unsigned char *d_sobel_x_array;
    unsigned char *d_sobel_y_array;

    size_t patch_size = nb_patch_x * nb_patch_y * sizeof(unsigned char);
    size_t img_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_patch_x, patch_size);
    cudaMalloc(&d_patch_y, patch_size);
    cudaMalloc(&d_sobel_x_array, img_size);
    cudaMalloc(&d_sobel_y_array, img_size);

    cudaMemcpy( d_patch_x, img_sobel_patch_x_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_patch_y, img_sobel_patch_y_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_sobel_x_array, img_sobel_x_array, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_sobel_y_array, img_sobel_y_array, img_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_patch<<<gridSize, blockSize>>>(d_patch_x, d_patch_y, d_sobel_x_array,
            d_sobel_y_array, pool_size, width, height, nb_patch_x, nb_patch_y,
            blockSize, gridSize);

    cudaDeviceSynchronize();

    cudaMemcpy(img_sobel_patch_x_array, d_patch_x, patch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_sobel_patch_y_array, d_patch_y, patch_size, cudaMemcpyDeviceToHost);


}


void Image::create_response_array()
{
    unsigned char *d_patch_x;
    unsigned char *d_patch_y;
    unsigned char *d_response;

    size_t patch_size = nb_patch_x * nb_patch_y * sizeof(unsigned char);

    cudaMalloc(&d_patch_x, patch_size);
    cudaMalloc(&d_patch_y, patch_size);
    cudaMalloc(&d_response, patch_size);

    cudaMemcpy( d_patch_x, img_sobel_patch_x_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_patch_y, img_sobel_patch_y_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_response, img_response_array, patch_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_response<<<gridSize, blockSize>>>(d_response, d_patch_x, d_patch_y,
            nb_patch_x, nb_patch_y, blockSize, gridSize);

    cudaDeviceSynchronize();

    cudaMemcpy(img_response_array, d_response, patch_size, cudaMemcpyDeviceToHost);
}


void Image::create_response_clean_array()
{
    unsigned char *d_response;
    unsigned char *d_response_clean_1;
    unsigned char *d_response_clean_2;

    size_t patch_size = nb_patch_x * nb_patch_y * sizeof(unsigned char);


    cudaMalloc(&d_response, patch_size);
    cudaMalloc(&d_response_clean_1, patch_size);
    cudaMalloc(&d_response_clean_2, patch_size);


    cudaMemcpy( d_response, img_response_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_response_clean_1, img_response_clean_1_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_response_clean_2, img_response_clean_2_array, patch_size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 5;
    gridSize = 2;

    compute_dilatation<<<gridSize, blockSize>>>(d_response, d_response_clean_1,
            nb_patch_x, nb_patch_y, blockSize, gridSize);

    compute_erosion<<<gridSize, blockSize>>>(d_response_clean_1, d_response_clean_2,
            nb_patch_x, nb_patch_y, blockSize, gridSize);

    cudaMemcpy(img_response_clean_1_array, d_response_clean_1, patch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_response_clean_2_array, d_response_clean_2, patch_size, cudaMemcpyDeviceToHost);

}


int Image::get_size()
{
    return width * height;
}


int main(void)
{
    Image image("../../img/codebar.jpg", 31);
    image.print_image();

    image.create_gray_array();
    image.save_gray_img();

    image.create_sobel_array();
    image.save_sobel_img();

    image.create_patch_array();
    image.save_patch_img();

    image.create_response_array();
    image.save_response_img();

    //image.create_response_clean_array();
    image.save_response_clean_img();

    return 0;
}
