#include "gpu_functions.cuh"

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

        if (id < 2 * width)
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
                for (int j = -2; j <= 2; j++)
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
                for (int j = -2; j <= 2; j++)
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

__global__ void compute_final(unsigned char *d_final,
                              unsigned char *d_response_clean_2, int nb_patch_x, int nb_patch_y,
                              int width, int weight, int blockSize, int gridSize, int pool_size)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = nb_patch_x * nb_patch_y;

    while (id < img_size)
    {
        int value = d_response_clean_2[id];

        int id_patch_x = id % nb_patch_x;
        int id_patch_y = id / nb_patch_x;

        int id_pixel = pool_size * id_patch_y * width;
        id_pixel += id_patch_x * pool_size;

        for (int i = 0; i < pool_size; i++)
        {
            for (int j = 0; j < pool_size; j++)
            {
                d_final[id_pixel + i * width + j] = value;
            }
        }

        id += blockSize * gridSize;
    }
}



__global__ void compute_threshold(unsigned char* d_response_clean_2, int max_value, 
                                   int nb_patch_x, int nb_patch_y, int nb_block, int nb_thread);
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = nb_patch_x * nb_patch_y;
  
    while (id < img_size)
    {
      
        if (d_response_clean_2[id] < (0.5 * max_value))
        {
            d_response_clean_2[id] = 0;
        }
      
        id += blockSize * gridSize;
    }
  
}
