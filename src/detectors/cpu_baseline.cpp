#include "cpu_baseline.hpp"


CPUBaseline::CPUBaseline(DetectorMode mode)
    : DetectorInterface(mode){
    }


void CPUBaseline::compute_derivatives(Mat img, int pool_size, int n_filters) {
    if (n_filters != 2 || !pool_size) {
        std::cerr << "Error: compute derivatives with more than 2 filters"
            <<" isn't implemented yet." << std::endl;
        exit(1);
    }

    Mat h_derivatives = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);
    Mat v_derivatives = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);

    Mat sobel_x = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    Mat patch;

    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            patch = img(Range(i - 1,i + 2), Range(j - 1,j + 2));
            patch.convertTo(patch, CV_32FC1);


            h_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_x * patch)[0]);
            v_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_y * patch)[0]);
        }
    }

    this->h_derivatives_ = h_derivatives;
    this->v_derivatives_ = v_derivatives;

    if (true) {
        std::string win_name = std::string("horizontal derivative image");
        namedWindow(win_name);
        imshow(win_name, this->h_derivatives_);
        waitKey(0);
        destroyWindow(win_name);

        std::string win2_name = std::string("vertical derivative image");
        namedWindow(win2_name);
        imshow(win2_name, this->v_derivatives_);
        waitKey(0);
        destroyWindow(win2_name);
    }
}
