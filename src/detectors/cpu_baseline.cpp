#include "cpu_baseline.hpp"


CPUBaseline::CPUBaseline(DetectorMode mode)
    : DetectorInterface(mode) {
    }


void CPUBaseline::compute_derivatives(int pool_size, int n_filters) {
    if (n_filters != 2 || !pool_size) {
        std::cerr << "Error: compute derivatives with more than 2 filters"
            <<" isn't implemented yet." << std::endl;
        exit(1);
    }

    Mat h_derivatives = Mat::zeros(Size(this->img_.cols, this->img_.rows), CV_8UC1);
    Mat v_derivatives = Mat::zeros(Size(this->img_.cols, this->img_.rows), CV_8UC1);

    Mat sobel_x = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobel_y = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    Mat patch;

    for (int i = 1; i < this->img_.rows - 1; i++) {
        for (int j = 1; j < this->img_.cols - 1; j++) {
            patch = this->img_(Range(i - 1,i + 2), Range(j - 1,j + 2));
            patch.convertTo(patch, CV_32FC1);


            h_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_x * patch)[0]);
            flip(patch, patch, 0);
            transpose(patch, patch);
            v_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_x * patch)[0]);
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


void CPUBaseline::compute_gradient(int pool_size) {
    Mat h_patch = Mat::zeros(Size(this->img_.cols / pool_size, this->img_.rows / pool_size), CV_8UC1);
    Mat v_patch = Mat::zeros(Size(this->img_.cols / pool_size, this->img_.rows / pool_size), CV_8UC1);

    int x_patch_begin, x_patch_end, y_patch_begin, y_patch_end;

    Mat tmp_patch;
    for (int i = 0; i < (this->img_.rows / pool_size); i++) {
        x_patch_begin = i * pool_size;
        x_patch_end = (i + 1) * pool_size;

        for (int j = 0; j < (this->img_.cols / pool_size); j++) {
            y_patch_begin = j * pool_size;
            y_patch_end = (j + 1) * pool_size;

            tmp_patch = this->h_derivatives_(Range(x_patch_begin, x_patch_end), Range(y_patch_begin, y_patch_end));
            h_patch.at<unsigned char>(i, j) = mean(tmp_patch)[0];

            tmp_patch = this->v_derivatives_(Range(x_patch_begin, x_patch_end), Range(y_patch_begin, y_patch_end));
            v_patch.at<unsigned char>(i, j) = mean(tmp_patch)[0];
            
        }
    }

    this->h_patch_gradient_ = h_patch;
    this->v_patch_gradient_ = v_patch;
}


void CPUBaseline::load_img(std::string path, int scale) {
    this->DetectorInterface::load_img(path, scale);
    this->img_ = this->DetectorInterface::get_img();
}
