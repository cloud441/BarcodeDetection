#include "image.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#include <iostream>

__global__ void compute_gray(Image img, int blockSize, int gridSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int img_size = img.get_size();

	while (id < img_size)
	{
		c[id] = a[id] + b[id];
		printf("I'm %d, %d and i compute the element %d\n", blockIdx.x, threadIdx.x, id);
		id += blockSize * gridSize;
	}

}



Image::Image(int width_arg, int height_arg, int nb_chan_arg)
{
	width = width_arg;
	height = height_arg;
	nb_chan = nb_chan_arg;

	img_gray_array = (unsigned char *) malloc(width * height * sizeof(unsigned int));
}

Image::Image(const char* path)
{
	img_array = stbi_load(path, &width, &height, &nb_chan, 0);

	if (img_array == NULL)
	{
		std::cout << "Error : can't open the image: " << path << "\n";
	}
}

Image::~Image()
{
	if (img_array)
	{
		stbi_image_free(img_array);
	}

	free(img_gray_array);
}

void create_gray_array()
{

}

int Image::get_size()
{
	return width * height;
}
