#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"



class Image {

    public:

        Image(const char* path, int pool_size);
        ~Image();

        void create_gray_array();
        void create_sobel_array();
        void create_patch_array();
        void create_response_array();
        void create_response_clean_array();

        void save_gray_img();
        void save_sobel_img();
        void save_patch_img();
        void save_response_img();
        void save_response_clean_img();

        void print_image();
        int get_size();

    private:

        int width;
        int height;
        int nb_chan;
        int pool_size;
        int nb_patch_x;
        int nb_patch_y;


        unsigned char *img_array;
        unsigned char *img_gray_array;
        unsigned char *img_sobel_x_array;
        unsigned char *img_sobel_y_array;
        unsigned char *img_sobel_patch_x_array;
        unsigned char *img_sobel_patch_y_array;
        unsigned char *img_response_array;
        unsigned char *img_response_clean_1_array;
        unsigned char *img_response_clean_2_array;
};
