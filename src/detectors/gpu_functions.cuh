__global__ void compute_gray(unsigned char *d_gray_array,
    unsigned char *d_array, int width, int weight,
    int blockSize, int gridSize);

__global__ void compute_sobel(unsigned char *d_sobel_x,
    unsigned char *d_sobel_y, unsigned char *d_gray_array, int width,
    int weight, int blockSize, int gridSize);

__global__ void compute_patch(unsigned char *d_patch_x, unsigned char *d_patch_y,
    unsigned char *d_sobel_x_array, unsigned char *d_sobel_y_array,
    int pool_size, int width, int height, int nb_patch_x, int nb_patch_y,
    int blockSize, int gridSize);

__global__ void compute_response(unsigned char *d_response,
        unsigned char *d_patch_x, unsigned char *d_patch_y,
        int nb_patch_x, int nb_patch_y, int blockSize, int gridSize);

__global__ void compute_dilatation(unsigned char *d_response,
            unsigned char *d_response_clean_1,
            int width, int weight, int blockSize, int gridSize);

__global__ void compute_erosion(unsigned char *d_response_clean_1,
                unsigned char *d_response_clean_2,
                int width, int weight, int blockSize, int gridSize);

__global__ void compute_final(unsigned char *d_final,
                    unsigned char *d_response_clean_2, int nb_patch_x, int nb_patch_y,
                    int width, int weight, int blockSize, int gridSize, int pool_size);