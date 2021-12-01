#include "detectorInterface.hpp"

class GPUBaseline: public DetectorInterface {

private:
    bool display_;
    DetectorMode mode_;

    cv::cuda::GpuMat img_;

public:
    GPUBaseline(DetectorMode mode, bool display=false);

    void compute_derivatives(int pool_size = 31, int n_filters = 2);
    void compute_gradient(int pool_size = 31);
    void compute_barcodeness();
    void clean_barcodeness(int pp_pool_size = 5);

    void show_final_result(int pool_size = 31);

    __host__ void load_img(std::string path, int scale = 1);

};