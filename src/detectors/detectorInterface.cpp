#include "detectorInterface.hpp"
#include <ctime>
#include <chrono>

DetectorInterface::DetectorInterface(DetectorMode mode, bool display) {
    this->mode_ = mode;
    this->display_ = display;
}


void DetectorInterface::load_img(std::string path, int scale) {
    this->img_ = imread(path, IMREAD_GRAYSCALE);

    if (this->img_.empty()) {
        std::cerr << "Error: impossible to load image: " << path << std::endl;
        exit(1);
    }

    for (int i = 0; i < scale; i++) {
        pyrDown(this->img_, this->img_);
    }

    if (display_) {
        std::string win_name = std::string("Grayscale and scaled image");
        namedWindow(win_name);
        imshow(win_name, this->img_);
        waitKey(0);
        destroyWindow(win_name);
    }
}

Mat DetectorInterface::get_img() {
    return this->img_;
}

void  DetectorInterface::gpu_benchmark_start()
{
}

void  DetectorInterface::gpu_benchmark_end()
{
}

void  DetectorInterface::cpu_benchmark_start()
{
    this->clock_ = std::chrono::high_resolution_clock::now();
}

void  DetectorInterface::cpu_benchmark_end()
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - clock_).count();
    printf ("CPU: %f tick_ms (%f seconds).\n", duration, (float) duration / 1000.0);

}
