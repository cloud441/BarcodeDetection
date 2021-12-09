#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"



class Image {

    public:

        Image(const char* path);
        ~Image();

        void create_gray_array();
        void create_sobel_array();

        void save_gray_img();
        void save_sobel_img();

        int get_size();

    private:

        int width;
        int height;
        int nb_chan;

        unsigned char *img_array;
        unsigned char *img_gray_array;
        unsigned char *img_sobel_x_array;
        unsigned char *img_sobel_y_array;

};
