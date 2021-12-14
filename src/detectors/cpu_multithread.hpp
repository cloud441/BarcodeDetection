#pragma once

#include <cstdlib>
#include <thread>
#include <vector>
#include <cmath>

#include "detectorInterface.hpp"
#include "utils.hpp"


class CPUMultithread: public DetectorInterface {

    private:

        bool display_;
        DetectorMode mode_;
        Mat img_;

        Mat h_derivatives_;
        Mat v_derivatives_;
        Mat h_patch_gradient_;
        Mat v_patch_gradient_;
        Mat patch_barcodeness_;
        Mat final_result_;

    public:

        CPUMultithread(DetectorMode mode, bool display=false);
        void compute_derivatives(int pool_size = 31, int n_filters = 2);
        void compute_gradient(int pool_size = 31);
        void compute_barcodeness();
        void clean_barcodeness(int pp_pool_size = 5);

        void show_final_result(int pool_size = 31);

        void load_img(std::string path, int scale = 1);

    private:

        static void thread_compute_derivatives(Mat sub_img, int index, std::vector<Mat> *derivatives_vec, int pool_size, int n_filters);
        static void thread_compute_gradient(Mat h_sub_img, Mat v_sub_img, int index, std::vector<Mat> *derivatives_vec, int pool_size);
        static void thread_compute_barcodeness(Mat h_sub_gradient, Mat v_sub_gradient, int index, std::vector<Mat> *patch_barcodeness_vec);
        static void thread_clean_barcodeness(Mat sub_mat, int index, std::vector<Mat> *clean_blocks_vec, Mat struct_element);
};
