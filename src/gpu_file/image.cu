#include "image.hh"

#include <iostream>

__global__ void compute_gray(unsigned char *d_gray_array,
            unsigned char *d_array, int img_width, int img_weight,
            int blockSize, int gridSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = img_width * img_weight;

    while (id < img_size)
    {
        d_gray_array[id] = d_array[id * 3] * 0.2989 +
            d_array[id * 3 + 1] * 0.5870 + d_array[id * 3 + 2] * 0.1140;

        id += blockSize * gridSize;
    }

}



Image::Image(const char* path)
{

    img_array = stbi_load(path, &width, &height, &nb_chan, 0);

    if (img_array == NULL)
    {
        std::cout << "Error : can't open the image: " << path << "\n";
    }

    img_gray_array = new unsigned char[width * height];
}

Image::~Image()
{
    if (img_array)
    {
        stbi_image_free(img_array);
    }

    free(img_gray_array);
}

void Image::save_gray_img()
{
    stbi_write_jpg("../../img/codebar_gray_result.jpg", width, height, nb_chan, img_array, 100);
}

void Image::create_gray_array()
{

    // CPU Version
    for (int i = 0; i < height * width; ++i)
    {
        img_gray_array[i] = img_array[i * 3] * 0.2989 +
            img_array[i * 3 + 1] * 0.5870 + img_array[i * 3 + 2] * 0.1140;
    }

}

int Image::get_size()
{
    return width * height;
}


int main(void)
{
    Image image("../../img/codebar.jpg");

    image.create_gray_array();

    image.save_gray_img();

    return 0;
}
