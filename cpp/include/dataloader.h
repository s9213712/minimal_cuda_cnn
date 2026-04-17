#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "tensor.h"

class Cifar10Loader {
    std::string data_dir;
public:
    Cifar10Loader(std::string dir) : data_dir(dir) {}

    // Simplified loader for the sake of prototype: 
    // In real CIFAR-10, data is in binary files (data_batch_1, etc.)
    // Here we implement a basic binary reader that loads images into a CudaTensor
    void load_batch(int batch_size, CudaTensor* target) {
        std::string file_path = data_dir + "/data_batch_1";
        std::ifstream file(file_path, std::ios::binary);
        
        if (!file.is_open()) {
            std::cerr << "[Error] Could not open CIFAR-10 data file: " << file_path << std::endl;
            return;
        }

        // CIFAR-10 binary format: 1 byte label, 3072 bytes image (3*32*32)
        std::vector<float> host_data(batch_size * 3 * 32 * 32);
        for (int i = 0; i < batch_size; ++i) {
            char label;
            file.read(&label, 1);
            for (int j = 0; j < 3 * 32 * 32; ++j) {
                unsigned char val;
                file.read(&val, 1);
                host_data[i * 3 * 32 * 32 + j] = (float)val / 255.0f;
            }
        }
        target->copy_from_host(host_data.data());
    }
};
