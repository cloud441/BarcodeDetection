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


std::vector<Mat> concatenate_h_v_block(std::vector<Mat> blocks, int n_blocks) {
    Mat h_final_mat;
    Mat v_final_mat;
    Mat h_line_matrix;
    Mat v_line_matrix;
    Mat h_cur_mat;
    Mat v_cur_mat;

    std::vector<Mat> final_imgs(2);

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
                    h_final_mat = h_line_matrix;
                    v_final_mat = v_line_matrix;
                }
                else {
                    vconcat(h_final_mat, h_line_matrix, h_final_mat);
                    vconcat(v_final_mat, v_line_matrix, v_final_mat);
                }
            }
        }
    }

    final_imgs[0] = h_final_mat;
    final_imgs[1] = v_final_mat;

    return final_imgs;
}

Mat concatenate_block(std::vector<Mat> blocks, int n_blocks) {
    Mat final_mat;
    Mat line_matrix;
    Mat cur_mat;

    int x_division_nb = sqrt(n_blocks);
    while (n_blocks % x_division_nb)
        x_division_nb -= 1;
    int y_division_nb = n_blocks / x_division_nb;


    for (int i = 0; i < x_division_nb; i++) {
        for (int j = 0; j < y_division_nb; j++) {

            cur_mat = blocks[i * y_division_nb + j];

            if (!j) {
                line_matrix = cur_mat;
            }
            else {
                hconcat(line_matrix, cur_mat, line_matrix);
            }

            if (j == y_division_nb - 1) {
                if (i == 0) {
                    final_mat = line_matrix;
                }
                else {
                    vconcat(final_mat, line_matrix, final_mat);
                }
            }
        }
    }
    
    return final_mat;
}
