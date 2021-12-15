#include "gpu_baseline.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <../stb_image/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <../stb_image/stb_image_write.h>


GPUBaseline::GPUBaseline(int block, int thread) {
    nb_block = block;
    nb_thread = thread;
    std::filesystem::create_directory(out_dir_);
}

void GPUBaseline::load_img(std::string path, int pool_size_arg)
{
    img_path_ = path;
    img_fname_ = std::filesystem::path(img_path_).replace_extension().filename();
    img_array = stbi_load(path.c_str(), &width, &height, &nb_chan, 0);

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
    final_img = new unsigned char[width * height];
}

GPUBaseline::~GPUBaseline()
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
    free(final_img);
}


void GPUBaseline::print_image()
{
    printf("The image have a size of %d x %d x %d\n", width, height, nb_chan);
    printf("The pool size is %d\n", pool_size);
    printf("And so the image have %d patch on x and %d patch en y\n", nb_patch_x, nb_patch_y);
}

void GPUBaseline::save_gray_img()
{
    auto f_path = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_gray.jpg");
    stbi_write_jpg(f_path.c_str(), width, height, 1,
            img_gray_array, 100);
}

void GPUBaseline::save_sobel_img()
{
    auto f_path_x = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_sobel_x.jpg");
    stbi_write_jpg(f_path_x.c_str(), width, height, 1,
            img_sobel_x_array, 100);
    auto f_path_y = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_sobel_y.jpg");
    stbi_write_jpg(f_path_y.c_str(), width, height, 1,
            img_sobel_y_array, 100);
}

void GPUBaseline::save_patch_img()
{
    auto f_path_x = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "patch_x.jpg");
    stbi_write_jpg(f_path_x.c_str(), nb_patch_x, nb_patch_y, 1,
        img_sobel_patch_x_array, 100);
    auto f_path_y = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "patch_y.jpg");
    stbi_write_jpg(f_path_y.c_str(), nb_patch_x, nb_patch_y, 1,
        img_sobel_patch_y_array, 100);

}

void GPUBaseline::save_response_img()
{
    auto f_path = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_response.jpg");
    stbi_write_jpg(f_path.c_str(), nb_patch_x, nb_patch_y, 1,
        img_response_array, 100);
}

void GPUBaseline::save_response_clean_img()
{
    auto f_path = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_response_clean.jpg");
    stbi_write_jpg(f_path.c_str(), nb_patch_x, nb_patch_y, 1,
        img_response_clean_2_array, 100);
}

void GPUBaseline::save_final()
{
    auto f_path = std::filesystem::path(out_dir_) / std::filesystem::path(img_fname_ + "_final.jpg");
    stbi_write_jpg(f_path.c_str(), width, height, 1,
        final_img, 100);
}


void GPUBaseline::create_gray_array()
{

    unsigned char *d_gray_img;
    unsigned char *d_img;

    size_t gray_img_size = width * height * sizeof(unsigned char);
    size_t img_size = width * height * 3 * sizeof(unsigned char);

    cudaMalloc(&d_gray_img, gray_img_size);
    cudaMalloc(&d_img, img_size);


    cudaMemcpy( d_gray_img, img_gray_array, gray_img_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_img, img_array, img_size, cudaMemcpyHostToDevice);

    compute_gray<<<nb_block, nb_thread>>>(d_gray_img, d_img, width, height,
                                            nb_block, nb_thread);
    cudaDeviceSynchronize();

    cudaMemcpy(img_gray_array, d_gray_img, gray_img_size, cudaMemcpyDeviceToHost);

}

void GPUBaseline::create_sobel_array()
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

    compute_sobel<<<nb_block, nb_thread>>>(d_sobel_x, d_sobel_y, d_gray_img,
                                        width, height, nb_block, nb_thread);
    cudaDeviceSynchronize();

    cudaMemcpy(img_sobel_x_array, d_sobel_x, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_sobel_y_array, d_sobel_y, img_size, cudaMemcpyDeviceToHost);
}

void GPUBaseline::create_patch_array()
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

    compute_patch<<<nb_block, nb_thread>>>(d_patch_x, d_patch_y, d_sobel_x_array,
            d_sobel_y_array, pool_size, width, height, nb_patch_x, nb_patch_y,
            nb_block, nb_thread);

    cudaDeviceSynchronize();

    cudaMemcpy(img_sobel_patch_x_array, d_patch_x, patch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_sobel_patch_y_array, d_patch_y, patch_size, cudaMemcpyDeviceToHost);


}

void GPUBaseline::create_response_array()
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

    compute_response<<<nb_block, nb_thread>>>(d_response, d_patch_x, d_patch_y,
            nb_patch_x, nb_patch_y, nb_block, nb_thread);

    cudaDeviceSynchronize();

    cudaMemcpy(img_response_array, d_response, patch_size, cudaMemcpyDeviceToHost);
}

void GPUBaseline::create_response_clean_array()
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

    compute_dilatation<<<nb_block, nb_thread>>>(d_response, d_response_clean_1,
            nb_patch_x, nb_patch_y, nb_block, nb_thread);

    compute_erosion<<<nb_block, nb_thread>>>(d_response_clean_1, d_response_clean_2,
            nb_patch_x, nb_patch_y, nb_block, nb_thread);


    cudaMemcpy(img_response_clean_1_array, d_response_clean_1, patch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_response_clean_2_array, d_response_clean_2, patch_size, cudaMemcpyDeviceToHost);
    
    int max_value = 0;
    
    for (int i = 0; i < nb_patch_x * nb_patch_y; i++)
    {
        if (img_response_clean_2_array[i] > max_value)
        {
            max_value = img_response_clean_2_array[i];
        }
    }
    
    compute_threshold<<<nb_block, nb_thread>>>(d_response_clean_2, max_value, 
                                   nb_patch_x, nb_patch_y, nb_thread, nb_block);

    cudaMemcpy(img_response_clean_2_array, d_response_clean_2, patch_size, cudaMemcpyDeviceToHost);
}


void GPUBaseline::create_final()
{
    unsigned char *d_response_clean_2;
    unsigned char *d_final;

    size_t patch_size = nb_patch_x * nb_patch_y * sizeof(unsigned char);
    size_t img_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_response_clean_2, patch_size);
    cudaMalloc(&d_final, img_size);

    cudaMemcpy( d_response_clean_2, img_response_clean_2_array, patch_size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_final, final_img, img_size, cudaMemcpyHostToDevice);

    compute_final<<<nb_block, nb_thread>>>(d_final, d_response_clean_2,
            nb_patch_x, nb_patch_y, width, height,
            nb_block, nb_thread, pool_size);

    cudaMemcpy(final_img, d_final, img_size, cudaMemcpyDeviceToHost);

}


int GPUBaseline::get_size()
{
    return width * height;
}
