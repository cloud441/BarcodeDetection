#include "gpu_basline.hpp"

GPUBaseline::GPUBaseline(DetectorMode mode, bool display)
    : DetectorInterface(mode, display)
{}

__host__ void GPUBaseline::load_img(std::string path, int scale = 1) {
    this->DetectorInterface::load_img(path, scale);
    // Envoie l'image depuis le host vers le device
    this->img_.upload(this->DetectorInterface::get_img());
    std::cout << this->img_.size() << "\n"; 
}   