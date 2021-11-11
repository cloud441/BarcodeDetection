#pragma once

#include "detectorInterface.hpp"


class CPUBaseline: public DetectorInterface {

    public:

        CPUBaseline(DetectorMode mode);
        void compute_derivatives(Mat img, int pool_size, int n_filters = 2);
};
