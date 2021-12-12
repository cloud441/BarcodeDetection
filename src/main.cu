#include "main.hpp"

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Error: you need to provide a image path (1 argument required)" << std::endl;
        exit(1);
    }
    GPUBaseline detector = GPUBaseline();
    detector.load_img(argv[1]);

    detector.print_image();

    detector.create_gray_array();
    detector.save_gray_img();

    detector.create_sobel_array();
    detector.save_sobel_img();

    detector.create_patch_array();
    detector.save_patch_img();

    detector.create_response_array();
    detector.save_response_img();

    detector.create_response_clean_array();
    detector.save_response_clean_img();

    detector.create_final();
    detector.save_final();

    return 0;
}
