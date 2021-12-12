#include <string>
#include <iostream>

#include "gpu_functions.cuh"



class GPUBaseline
{

private:
    bool display_;

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
    unsigned char *final_img;

public:
    GPUBaseline();
    ~GPUBaseline();

    void load_img(std::string path, int pool_size_arg = 1);
    void create_gray_array();
    void create_sobel_array();
    void create_patch_array();
    void create_response_array();
    void create_response_clean_array();
    void create_final();

    void save_gray_img();
    void save_sobel_img();
    void save_patch_img();
    void save_response_img();
    void save_response_clean_img();
    void save_final();

    void print_image();
    int get_size();
};