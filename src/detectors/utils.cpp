#include "utils.hpp"


Mat get_block_from_index(Mat img, int index, int index_max) {
    int x_division_nb = sqrt(index_max);
    int y_division_nb;

    while (index_max % x_division_nb)
        x_division_nb -= 1;

    y_division_nb = index_max / x_division_nb;

    int x_block_size = img.rows / x_division_nb;
    int y_block_size = img.cols / y_division_nb;

    int x_min = (index / y_division_nb) * x_block_size;
    int x_max = x_min + x_block_size;
    int y_min = (index % y_division_nb) * y_block_size;
    int y_max = y_min + y_block_size;

    return img(Range(x_min, x_max), Range(y_min, y_max));
}


std::vector<Mat> concatenate_derivatives_block(std::vector<Mat> blocks, int n_blocks) {
    Mat h_derivatives;
    Mat v_derivatives;
    Mat h_line_matrix;
    Mat v_line_matrix;
    Mat h_cur_mat;
    Mat v_cur_mat;

    std::vector<Mat> derivatives_img(2);

    int x_division_nb = sqrt(n_blocks);
    while (n_blocks % x_division_nb)
        x_division_nb -= 1;
    int y_division_nb = n_blocks / x_division_nb;


    for (int i = 0; i < x_division_nb; i++) {
        for (int j = 0; j < y_division_nb; j++) {
            h_cur_mat = blocks[2 * (i * y_division_nb + j)];
            v_cur_mat = blocks[2 * (i * y_division_nb + j) + 1];

            if (!j) {
                h_line_matrix = h_cur_mat;
                v_line_matrix = v_cur_mat;
            }
            else {
                hconcat(h_line_matrix, h_cur_mat, h_line_matrix);
                hconcat(v_line_matrix, v_cur_mat, v_line_matrix);
            }

            if (j == y_division_nb - 1) {
                if (i == 0) {
                    h_derivatives = h_line_matrix;
                    v_derivatives = v_line_matrix;
                }
                else {
                    vconcat(h_derivatives, h_line_matrix, h_derivatives);
                    vconcat(v_derivatives, v_line_matrix, v_derivatives);
                }
            }
        }
    }

    derivatives_img[0] = h_derivatives;
    derivatives_img[1] = v_derivatives;

    return derivatives_img;
}
