#include "detector.hpp"

int main(void)
{

    DetectorInterface detector = DetectorInterface(DetectorMode::IMAGE);
    detector.load_img("../data/bate.jpg");

    return 0;
}
