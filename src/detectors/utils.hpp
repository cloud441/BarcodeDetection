#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;


Mat get_block_from_index(Mat img, int index, int index_max);
std::vector<Mat> concatenate_h_v_block(std::vector<Mat> blocks, int n_blocks);
Mat concatenate_block(std::vector<Mat> blocks, int n_blocks);
