#ifndef NET_INIT_H
#define NET_INIT_H

#include <vector>

// Define a structure to hold the RBFNN parameters
struct RBFNN {
    std::vector<std::vector<double>> centers;  // Centers of the Gaussian functions (size: r x (d + k))
    std::vector<double> widths;                // Widths of the Gaussian functions (size: r)
    std::vector<std::vector<double>> weights;  // Weights for the output layer (size: r x t)
};

// Function declarations for initializing the RBFNN
RBFNN initialize_rbfnn(int r, int input_dim, const std::vector<std::vector<double>>& data);

#endif // NET_INIT_H
