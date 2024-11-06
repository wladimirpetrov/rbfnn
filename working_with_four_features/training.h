#ifndef TRAINING_H
#define TRAINING_H

#include "net_init.h"
#include <vector>

// Function declaration for the forward pass
std::vector<std::vector<double>> forward_pass(RBFNN& rbfnn, const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& F_values, double eta, double lambda, double tolerance, int max_iterations);

// Function to compute Gaussian activations for each mission
std::vector<double> compute_activations(const std::vector<double>& x_i, const RBFNN& rbfnn);

// Function to calculate Jacobian and Hessian for each mission
std::vector<double> compute_jacobian(const std::vector<double>& activations);
std::vector<std::vector<double>> compute_hessian(const std::vector<double>& jacobian);

// Function to compute the gradient
std::vector<double> compute_gradient(const std::vector<double>& jacobian, double error);

// Function to multiply inverse Hessian by gradient
std::vector<double> multiply_hessian_inv_by_gradient(const std::vector<std::vector<double>>& hessian_inv, const std::vector<double>& gradient);

// Function to multiply vector by the learning rate
std::vector<double> multiply_by_eta(const std::vector<double>& vec, double eta);

// Function to update weights
void update_weights(RBFNN& rbfnn, const std::vector<double>& weight_updates);

// Function to compute the inverse of the Hessian matrix
std::vector<std::vector<double>> compute_hessian_inverse(const std::vector<std::vector<double>>& hessian_reg);

double calculate_norm(const std::vector<double>& vec);

#endif // TRAINING_H
