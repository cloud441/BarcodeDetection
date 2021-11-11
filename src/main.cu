#include "main.hpp"

int main(void)
{

    CPUBaseline detector = CPUBaseline(DetectorMode::IMAGE);
    detector.load_img("../data/bate.jpg");

    detector.compute_derivatives(detector.get_img(), 31);

    return 0;
}
