#include "cpu_multithread.hpp"


CPUMultithread::CPUMultithread(DetectorMode mode, bool display)
    : DetectorInterface(mode, display) {
        this->mode_ = mode;
        this->display_ = display;
    }


void CPUMultithread::thread_compute_derivatives(Mat sub_img, int index, std::vector<Mat> *derivatives_vec, int pool_size, int n_filters) {
    n_filters = n_filters;
    pool_size = pool_size;
    Mat h_derivatives = Mat::zeros(Size(sub_img.cols, sub_img.rows), CV_8UC1);
    Mat v_derivatives = Mat::zeros(Size(sub_img.cols, sub_img.rows), CV_8UC1);

    Mat sobel_x = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //Mat sobel_y = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    Mat patch;

    for (int i = 1; i < sub_img.rows - 1; i++) {
        for (int j = 1; j < sub_img.cols - 1; j++) {
            patch = sub_img(Range(i - 1,i + 2), Range(j - 1,j + 2));
            patch.convertTo(patch, CV_32FC1);


            h_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_x * patch)[0]);
            flip(patch, patch, 0);
            transpose(patch, patch);
            v_derivatives.at<unsigned char>(i, j) = std::abs(sum(sobel_x * patch)[0]);
        }
    }

    (*derivatives_vec)[2 * index] = h_derivatives;
    (*derivatives_vec)[2 * index + 1] = v_derivatives;
}



void CPUMultithread::compute_derivatives(int pool_size, int n_filters) {
    if (n_filters != 2 || !pool_size) {
        std::cerr << "Error: compute derivatives with more than 2 filters"
            <<" isn't implemented yet." << std::endl;
        exit(1);
    }

    // Split image in n_thread blocks to compute derivatives in multithreading:
    int n_thread =  std::thread::hardware_concurrency();
    std::vector<std::thread> thread_vec(n_thread);
    std::vector<Mat> derivatives_vec(2 * n_thread);

    for (auto it = std::begin(thread_vec); it != std::end(thread_vec); it++) {
        Mat sub_mat = get_block_from_index(this->img_, it - std::begin(thread_vec), n_thread);
        *it = std::thread(this->thread_compute_derivatives, sub_mat, it - std::begin(thread_vec), &derivatives_vec, pool_size, n_filters);
    }

    for (int i = 0; i < n_thread; i++) {
        thread_vec[i].join();
    }


    // Concatenate blocks to build original image:
    std::vector<Mat> derivatives = concatenate_derivatives_block(derivatives_vec, n_thread);

    this->h_derivatives_ = derivatives[0];
    this->v_derivatives_ = derivatives[1];

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


void CPUMultithread::compute_gradient(int pool_size) {
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


void CPUMultithread::load_img(std::string path, int scale) {
    this->DetectorInterface::load_img(path, scale);
    this->img_ = this->DetectorInterface::get_img();

    int n_thread = std::thread::hardware_concurrency();
    int x_division_nb = sqrt(n_thread);
    int y_division_nb;

    while (n_thread % x_division_nb)
        x_division_nb -= 1;

    y_division_nb = n_thread / x_division_nb;

    int x_block_size = this->img_.rows / x_division_nb;
    int y_block_size = this->img_.cols / y_division_nb;

    resize(this->img_, this->img_, Size(y_block_size * y_division_nb, x_block_size * x_division_nb), INTER_LINEAR);
}



void CPUMultithread::compute_barcodeness() {
    this->patch_barcodeness_ = this->v_patch_gradient_ - this->h_patch_gradient_;

    if (this->display_) {
        std::string win_name = std::string("patch barcodeness image");
        namedWindow(win_name);
        imshow(win_name, this->patch_barcodeness_);
        waitKey(0);
        destroyWindow(win_name);
    }
}


void CPUMultithread::clean_barcodeness(int pp_pool_size) {
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


void CPUMultithread::show_final_result(int pool_size) {
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
