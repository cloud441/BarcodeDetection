#include "detectorInterface.hpp"


DetectorInterface::DetectorInterface(DetectorMode mode) {
    this->mode_ = mode;
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

    if (true) {
        std::string win_name = std::string("Grayscale and scaled image");
        namedWindow(win_name);
        imshow(win_name, this->img_);
        waitKey(0);
        destroyWindow(win_name);
    }
}
