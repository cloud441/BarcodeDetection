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

        bool display_;
        DetectorMode mode_;
        Mat img_;


        Mat h_derivatives_;
        Mat v_derivatives_;
        Mat h_patch_gradient_;
        Mat v_patch_gradient_;
        Mat patch_barcodeness_;
        Mat final_result_;

        //benchmarck:
        std::clock_t clock_;




    public:

        DetectorInterface(DetectorMode mode, bool display = false);
        void load_img(std::string path, int scale = 1);

        void gpu_benchmark_start();
        void gpu_benchmark_end();
        void cpu_benchmark_start();
        void cpu_benchmark_end();

        virtual void compute_derivatives(int pool_size, int n_filters) = 0;
        virtual void compute_gradient(int pool_size) = 0;
        virtual void compute_barcodeness() = 0;
        virtual void clean_barcodeness(int pp_pool_size) = 0;
        virtual void show_final_result(int pool_size) = 0;

        Mat get_img();
};
