#pragma once

#include <cstdlib>

#include "detectorInterface.hpp"


class CPUBaseline: public DetectorInterface {

    private:

        Mat h_derivatives_;
        Mat v_derivatives_;

    public:

        CPUBaseline(DetectorMode mode);
        void compute_derivatives(Mat img, int pool_size, int n_filters = 2);
};
