#pragma once

#include <opencv2/opencv.hpp>
#include <string>


using namespace cv;


enum DetectorMode {
    IMAGE,
    VIDEO
};


class DetectorInterface {

    private:

        DetectorMode mode_;
        Mat img_;

        Mat h_derivatives_;
        Mat v_derivatives_;
        Mat h_patch_gradient_;
        Mat v_patch_gradient_;
        Mat patch_barcodeness_;



    public:

        DetectorInterface(DetectorMode mode);
        void load_img(std::string path, int scale = 1);

        virtual void compute_derivatives(int pool_size, int n_filters) = 0;
        virtual void compute_gradient(int pool_size) = 0;
        virtual void compute_barcodeness(int pool_size) = 0;


        Mat get_img();
};
