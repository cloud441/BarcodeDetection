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
    stbi_write_jpg("../../img/codebar_gray_result.jpg", width, height, 1, img_gray_array, 100);
}

void Image::create_gray_array()
{

    // CPU Version
    for (int i = 0; i < height * width; ++i)
    {
        /*
        img_gray_array[i] = img_array[i * 3] * 0.2989 +
            img_array[i * 3 + 1] * 0.5870 + img_array[i * 3 + 2] * 0.1140;
        */
        img_gray_array[i] = (uint8_t)((img_array[i * 3] + img_array[i * 3 + 1] + img_array[i * 3 + 2])/3.0);
    }

}

int Image::get_size()
{
    return width * height;
}


int main(void)
{
    //Image image("../../img/codebar.jpg");
    //image.create_gray_array();
    //image.save_gray_img();


    int width, height, nb_chan;
    unsigned char *img = stbi_load("../../img/codebar.jpg", &width, &height, &nb_chan, 0);

    size_t img_size = width * height * nb_chan;
    printf("We have a size of %dx%d, %d\n", width, height, img_size);
    int gray_chan = 1;
    size_t gray_img_size = width * height * gray_chan;

    unsigned char *gray_img = (unsigned char *) malloc(gray_img_size);

    for (unsigned char *p = img, *pg = gray_img; p != (img + img_size); p += nb_chan, pg += gray_chan)
    {
        *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
    }

    stbi_write_jpg("../../img/img_gray_result.jpg", width, height, gray_chan, gray_img, 100);

    return 0;
}
