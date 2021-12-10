#include "main.hpp"

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Error: you need to provide a image path (1 argument required)" << std::endl;
        exit(1);
    }
    /*
    CPUBaseline detector = CPUBaseline(DetectorMode::IMAGE);
    detector.load_img(argv[1], 2);

    detector.cpu_benchmark_start();
    detector.compute_derivatives(31);
    detector.cpu_benchmark_end();

    detector.compute_gradient(31);

    detector.compute_barcodeness();

    detector.clean_barcodeness();

    detector.show_final_result();*/

    GPUBaseline detector = GPUBaseline();
    detector.load_img(argv[1]);
    detector.create_gray_array();
    detector.save_gray_img();
    detector.compute_derivatives();
    detector.save_sobel_img();

    return 0;
}
