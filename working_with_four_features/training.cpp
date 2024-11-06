#include "training.h"
#include <cmath>
#include <iostream>

// Helper function to compute activations for each mission
std::vector<double> compute_activations(const std::vector<double>& x_i, const RBFNN& rbfnn) {
    std::vector<double> activations(rbfnn.centers.size(), 0.0);

    for (size_t j = 0; j < rbfnn.centers.size(); ++j) {
        double distance_squared = 0.0;

        // Compute the Euclidean distance squared between input x_i and center c_j
        for (size_t d = 0; d < x_i.size(); ++d) {
            distance_squared += std::pow(x_i[d] - rbfnn.centers[j][d], 2);
        }

        // Compute Gaussian activation h_j(x_i)
        activations[j] = std::exp(-distance_squared / (2 * std::pow(rbfnn.widths[j], 2)));

        // Print the computed activation for this neuron
        std::cout << "Activation of neuron " << j + 1 << " for mission input: " << activations[j] << std::endl;
    }

    return activations;
}

// Helper function to compute the Jacobian matrix
std::vector<double> compute_jacobian(const std::vector<double>& activations) {
    std::vector<double> jacobian(activations.size(), 0.0);

    for (size_t j = 0; j < activations.size(); ++j) {
        jacobian[j] = -activations[j];  // Jacobian is the negative of the activation
    }

    return jacobian;
}

// Helper function to compute the Hessian matrix (H = J^T * J)
std::vector<std::vector<double>> compute_hessian(const std::vector<double>& jacobian) {
    size_t size = jacobian.size();
    std::vector<std::vector<double>> hessian(size, std::vector<double>(size, 0.0));

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            hessian[i][j] = jacobian[i] * jacobian[j];  // H_ij = J_i * J_j
        }
    }

    return hessian;
}

// Helper function to compute the gradient g = J^T * e for each time step
std::vector<double> compute_gradient(const std::vector<double>& jacobian, double error) {
    std::vector<double> gradient(jacobian.size(), 0.0);

    // g_j = J_j * e for each j in the Jacobian
    for (size_t j = 0; j < jacobian.size(); ++j) {
        gradient[j] = jacobian[j] * error;
    }

    return gradient;
}

std::vector<double> multiply_hessian_inv_by_gradient(const std::vector<std::vector<double>>& hessian_inv, const std::vector<double>& gradient) {
    std::vector<double> result(gradient.size(), 0.0);

    for (size_t row = 0; row < gradient.size(); ++row) {
        for (size_t col = 0; col < gradient.size(); ++col) {
            result[row] += hessian_inv[row][col] * gradient[col];
        }
    }

    return result;
}

// Function to multiply by the learning rate η
std::vector<double> multiply_by_eta(const std::vector<double>& vec, double eta) {
    std::vector<double> result(vec.size(), 0.0);

    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = eta * vec[i];
    }

    return result;
}

// Function to update the weights
void update_weights(RBFNN& rbfnn, const std::vector<double>& weight_updates) {
    for (size_t i = 0; i < rbfnn.weights.size(); ++i) {
        for (size_t j = 0; j < rbfnn.weights[i].size(); ++j) {
            rbfnn.weights[i][j] += weight_updates[i];
        }
    }

    // Print updated weights
    std::cout << "Weights after update: [ ";
    for (const auto& weight : rbfnn.weights) {
        for (const auto& w : weight) {
            std::cout << w << " ";
        }
    }
    std::cout << "]\n";
}


// Function to compute the inverse of the regularized Hessian
std::vector<std::vector<double>> compute_hessian_inverse(const std::vector<std::vector<double>>& hessian_reg) {
    size_t size = hessian_reg.size();
    std::vector<std::vector<double>> hessian_inv(size, std::vector<double>(size, 0.0));

    if (size == 2) {
        double det = hessian_reg[0][0] * hessian_reg[1][1] - hessian_reg[0][1] * hessian_reg[1][0];
        if (det != 0) {
            hessian_inv[0][0] = hessian_reg[1][1] / det;
            hessian_inv[1][1] = hessian_reg[0][0] / det;
            hessian_inv[0][1] = -hessian_reg[0][1] / det;
            hessian_inv[1][0] = -hessian_reg[1][0] / det;
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            hessian_inv[i][i] = 1.0 / hessian_reg[i][i];
        }
    }

    return hessian_inv;
}
