#include "cpu_baseline.hpp"


CPUBaseline::CPUBaseline(DetectorMode mode, bool display)
    : DetectorInterface(mode, display) {
        this->mode_ = mode;
        this->display_ = display;
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

    if (this->display_) {
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



void CPUBaseline::compute_barcodeness() {
    this->patch_barcodeness_ = this->v_patch_gradient_ - this->h_patch_gradient_;    

    if (this->display_) {
        std::string win_name = std::string("patch barcodeness image");
        namedWindow(win_name);
        imshow(win_name, this->patch_barcodeness_);
        waitKey(0);
        destroyWindow(win_name);
    }
}


void CPUBaseline::clean_barcodeness(int pp_pool_size) {
    Mat struct_elt = getStructuringElement(MORPH_RECT, Size(pp_pool_size, pp_pool_size));

    struct_elt(Range(pp_pool_size / 2 - 1, pp_pool_size / 2 + 2), Range(0, pp_pool_size)) = 0;
    struct_elt = 1 - struct_elt;

    morphologyEx(this->patch_barcodeness_, this->patch_barcodeness_, MORPH_CLOSE, struct_elt);

    if (this->display_) {
        std::string win_name = std::string("cleaned patch barcodeness image");
        namedWindow(win_name);
        imshow(win_name, this->patch_barcodeness_);
        waitKey(0);
        destroyWindow(win_name);
    }
}


void CPUBaseline::show_final_result(int pool_size) {
    double max_value;
    minMaxLoc(this->patch_barcodeness_, nullptr, &max_value, nullptr, nullptr);

    threshold(this->patch_barcodeness_, this->patch_barcodeness_, max_value / 2, 255, THRESH_BINARY);
    resize(this->patch_barcodeness_, this->final_result_, Size(), pool_size, pool_size, INTER_NEAREST);
    
    if (this->display_) {
        std::string win_name = std::string("final barcode detection image");
        namedWindow(win_name);
        imshow(win_name, this->final_result_);
        waitKey(0);
        destroyWindow(win_name);
    }
}
