#include "cpu_baseline.hpp"


CPUBaseline::CPUBaseline(DetectorMode mode)
    : DetectorInterface(mode){
    }


void CPUBaseline::compute_derivatives(Mat img, int pool_size, int n_filters) {
    Mat h_derivatives = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);
    Mat v_derivatives = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);

    Mat sobel_x = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<double>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

}
