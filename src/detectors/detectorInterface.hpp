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



    public:

        DetectorInterface(DetectorMode mode);
        void load_img(std::string path, int scale = 1);

        //virtual Mat h_derivative(Mat img) = 0;
        //virtual Mat v_derivative(Mat img) = 0;

};
