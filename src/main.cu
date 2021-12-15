#include <string>
#include <stdio.h>
#include <iostream>

#include "main.hpp"

void print_help()
{
    std::cout << "Usage : ./detector <path> [option]\n\n"
        << "option : \n\n" << "\t--help : print the usage.\n"
        << "\t--cpu : use the cpu baseline.\n"
        << "\t--cpu_opti : use the multithread cpu baseline.\n"
        << "\t--gpu : use the gpu baseline.\n"
        << "\t--step : to save a .jpg of all step (only for gpu baseline).\n"
        << "\t--blocks <int> : number of block to use in the gpu baseline (only for gpu baseline).\n"
        << "\t--threads <int> : number of threads to use in the gpu baseline (only for gpu baseline).\n"
        << std::endl;
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        std::cerr << "Error: you need to provide a image path (1 argument required)" << std::endl;
        exit(1);
    }

    std::string path = std::string(argv[1]);

    bool cpu_opti = false;

    bool gpu = false;
    bool save_all_step = false;
    int nb_block = 5;
    int nb_thread = 20;


    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i],"--help") == 0)
        {
            print_help();
        }
        else if (strcmp(argv[i],"--cpu") == 0)
        {
            cpu_opti = false;
            gpu = false;
        }
        else if (strcmp(argv[i],"--cpu_opti") == 0)
        {
            cpu_opti = true;
            gpu = false;
        }
        else if (strcmp(argv[i],"--gpu") == 0)
        {
            cpu_opti = false;
            gpu = true;
        }
        else if (strcmp(argv[i],"--step") == 0)
        {
            save_all_step = true;
        }
        else if (strcmp(argv[i],"--blocks") == 0)
        {
            nb_block = std::stoi(argv[i + 1]);
        }
        else if (strcmp(argv[i],"--threads") == 0)
        {
            nb_thread = std::stoi(argv[i + 1]);
        }
    }

    if (gpu)
    {
        GPUBaseline detector = GPUBaseline(nb_block, nb_thread);
        detector.load_img(argv[1]);

        detector.create_gray_array();
        if (save_all_step)
        {
            detector.save_gray_img();
        }


        detector.create_sobel_array();
        if (save_all_step)
        {
            detector.save_sobel_img();
        }


        detector.create_patch_array();
        if (save_all_step)
        {
            detector.save_patch_img();
        }


        detector.create_response_array();
        if (save_all_step)
        {
            detector.save_response_img();
        }


        detector.create_response_clean_array();
        if (save_all_step)
        {
            detector.save_response_clean_img();
        }


        detector.create_final();
        detector.save_final();
    }

    else if (cpu_opti)
    {
        CPUMultithread detector = CPUMultithread(DetectorMode::IMAGE);
        detector.load_img(argv[1], 2);

        detector.cpu_benchmark_start();
        detector.compute_derivatives();
        detector.cpu_benchmark_end();
        detector.compute_gradient();

        detector.compute_barcodeness();

        detector.clean_barcodeness();

        detector.show_final_result();
    }

    else
    {
        CPUBaseline detector = CPUBaseline(DetectorMode::IMAGE);

        detector.cpu_benchmark_start();
        detector.load_img(argv[1], 2);

        detector.compute_derivatives();

        detector.compute_gradient();

        detector.compute_barcodeness();

        detector.clean_barcodeness();

        detector.show_final_result();
        detector.cpu_benchmark_end();

    }


    return 0;
}
