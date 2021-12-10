#include <string>
#include <iostream>

class GPUBaseline
{

private:
    bool display_;

    int width;
    int height;
    int nb_chan;

    unsigned char *img_array;
    unsigned char *img_gray_array;
    unsigned char *img_sobel_x_array;
    unsigned char *img_sobel_y_array;

public:
    GPUBaseline();

    void load_img(std::string path, int scale = 1);
    void create_gray_array();
    void compute_derivatives();

    void save_gray_img();
    void save_sobel_img();
    int get_size();
};