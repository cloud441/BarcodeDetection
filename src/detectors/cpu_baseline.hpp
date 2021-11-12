#pragma once

#include <cstdlib>

#include "detectorInterface.hpp"


class CPUBaseline: public DetectorInterface {

    private:

        DetectorMode mode_;
        Mat img_;

        Mat h_derivatives_;
        Mat v_derivatives_;
        Mat h_patch_gradient_;
        Mat v_patch_gradient_;
        Mat patch_barcodeness_;
        Mat final_result_;

    public:

        CPUBaseline(DetectorMode mode);
        void compute_derivatives(int pool_size = 31, int n_filters = 2);
        void compute_gradient(int pool_size = 31);
        void compute_barcodeness();
        void clean_barcodeness(int pp_pool_size = 5);

        void show_final_result(int pool_size = 31);

        void load_img(std::string path, int scale = 1);
};
