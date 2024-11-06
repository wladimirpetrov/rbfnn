#include "net_init.h"
#include "data_processing_output.h"  // Include the new header for output normalization
#include "training.h"
#include <iostream>
#include <fstream>
#include <cmath>

// Function to calculate the Euclidean norm of a vector (for error/gradient)
double calculate_norm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (const auto& val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Function to save the model parameters to a file
void save_model(const RBFNN& rbfnn, const std::string& filename, const std::vector<std::vector<std::vector<double>>>& updated_weights, int f_size) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file for saving the model." << std::endl;
        return;
    }

    // Save centers
    file << "Centers:\n";
    for (const auto& center : rbfnn.centers) {
        for (const auto& val : center) {
            file << val << ",";
        }
        file << "\n";
    }

    // Save widths
    file << "Widths:\n";
    for (const auto& width : rbfnn.widths) {
        file << width << ",";
    }
    file << "\n";

    // Save weights for each mission and time step (2 by f_size for each mission)
    file << "Weights:\n";
    for (size_t mission = 0; mission < updated_weights.size(); ++mission) {
        file << "Mission " << mission + 1 << " Weights:\n";
        for (size_t neuron = 0; neuron < updated_weights[mission].size(); ++neuron) {
            file << "Neuron " << neuron + 1 << ": ";
            for (int t = 0; t < f_size; ++t) {
                file << updated_weights[mission][neuron][t] << ",";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Model (centers, widths, and weights) saved to " << filename << std::endl;
}


std::vector<std::vector<double>> forward_pass(RBFNN& rbfnn, const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& F_values, double eta, double lambda, double tolerance, int max_iterations) {
    std::vector<std::vector<double>> predicted_trajectories(data.size());
    std::vector<std::vector<double>> errors(data.size());
    std::vector<std::vector<std::vector<double>>> gradients(data.size());

    // Initialize a 3D vector to store the updated weights for each mission and each time step
    std::vector<std::vector<std::vector<double>>> updated_weights(data.size(), std::vector<std::vector<double>>(rbfnn.weights.size(), std::vector<double>(F_values[0].size(), 0.0)));

    int iteration = 0;
    double error_norm = tolerance + 1.0;

    while (iteration < max_iterations && error_norm > tolerance) {
        error_norm = 0.0;
        std::cout << "\nIteration " << iteration + 1 << ":\n";

        for (size_t i = 0; i < data.size(); ++i) {  // Iterate over each mission
            const std::vector<double>& x_i = data[i];
            const std::vector<double>& F_i = F_values[i];

            std::vector<double> activations = compute_activations(x_i, rbfnn);

            // Adjust predicted trajectory calculation based on iteration number
            predicted_trajectories[i].resize(F_i.size(), 0.0);
            for (size_t t = 0; t < F_i.size(); ++t) {  // Iterate over each time step
                if (iteration == 0) {
                    // First iteration: Use initialized weights
                    for (size_t j = 0; j < rbfnn.centers.size(); ++j) {
                        predicted_trajectories[i][t] += rbfnn.weights[j][0] * activations[j];
                    }
                } else {
                    // Subsequent iterations: Use updated weights for each time step and mission
                    for (size_t j = 0; j < rbfnn.centers.size(); ++j) {
                        double current_weight = updated_weights[i][j][t];
                        double current_activation = activations[j];
                        double result = -current_weight * current_activation;
                                                // Print the current weight and activation to the terminal
                        std::cout << "Mission " << i + 1 << ", Time Step " << t + 1 << ", Neuron " << j + 1 << ": ";
                        std::cout << "Current Weight: " << current_weight << ", Current Activation: " << current_activation << std::endl;
                        

                        predicted_trajectories[i][t] += result;
                    }
                }
            }

            // Print the Error Trajectory (Difference between true and predicted trajectories)
            std::cout << "Mission " << i + 1 << " Error Trajectory: [";
            errors[i].resize(F_i.size(), 0.0);
            for (size_t t = 0; t < F_i.size(); ++t) {
                errors[i][t] = F_i[t] - predicted_trajectories[i][t];  // Element-wise difference
                std::cout << errors[i][t] << " ";  // Print the error at each time step
            }
            std::cout << "]\n";

            // Calculate errors and gradients
            errors[i].resize(F_i.size(), 0.0);
            gradients[i].resize(F_i.size());

            for (size_t t = 0; t < F_i.size(); ++t) {
                errors[i][t] = F_i[t] - predicted_trajectories[i][t];
                std::vector<double> jacobian = compute_jacobian(activations);
                std::vector<double> gradient = compute_gradient(jacobian, errors[i][t]);
                gradients[i][t] = gradient;

                std::vector<std::vector<double>> hessian = compute_hessian(jacobian);
                std::vector<std::vector<double>> identity(hessian.size(), std::vector<double>(hessian.size(), 0.0));
                for (size_t j = 0; j < hessian.size(); ++j) {
                    identity[j][j] = 1.0;
                }

                std::vector<std::vector<double>> hessian_reg(hessian.size(), std::vector<double>(hessian.size(), 0.0));
                for (size_t j = 0; j < hessian.size(); ++j) {
                    for (size_t k = 0; k < hessian.size(); ++k) {
                        hessian_reg[j][k] = hessian[j][k] + lambda * identity[j][k];
                    }
                }

                std::vector<std::vector<double>> hessian_inv = compute_hessian_inverse(hessian_reg);
                std::vector<double> hessian_gradient_result = multiply_hessian_inv_by_gradient(hessian_inv, gradient);
                std::vector<double> weight_updates = multiply_by_eta(hessian_gradient_result, eta);

                // Update weights for each time step and mission
                for (size_t j = 0; j < rbfnn.weights.size(); ++j) {
                    updated_weights[i][j][t] = rbfnn.weights[j][0] + weight_updates[j];  // Store updated weights
                }

                // Accumulate the norm of the error for convergence checking
                error_norm += calculate_norm(errors[i]);
            }
        }

        std::cout << "Total Error Norm: " << error_norm << std::endl;
        iteration++;
    }

    std::cout << "Training completed after " << iteration << " iterations." << std::endl;

    // Save the model (centers, widths, and weights)
    save_model(rbfnn, "trained_model.csv", updated_weights, F_values[0].size());

    return predicted_trajectories;
}

int main() {
    // Read the CSV data
    std::string filename = "data.csv";
    std::vector<Mission> missions = read_csv(filename);

    // Calculate mean and standard deviation for F_values (trajectories)
    std::vector<double> mean_F, std_F;
    calculate_mean_std_F(missions, mean_F, std_F);

    // Normalize F trajectories
    std::vector<std::vector<double>> normalized_F = normalize_F_trajectories(missions, mean_F, std_F);

    // Prepare data for RBFNN initialization
    std::vector<std::vector<double>> data;
    for (const auto& mission : missions) {
        data.push_back({mission.theta_0, mission.theta_cmd});
    }

    // Initialize the RBFNN
    int r = 2;  // Number of neurons
    int input_dim = 2;  // Features: theta_0 and theta_cmd
    RBFNN rbfnn = initialize_rbfnn(r, input_dim, data);

    // Training settings
    double eta = 10;  // Learning rate
    double lambda = 10;  // Regularization parameter
    double tolerance = 1e-4;  // Convergence tolerance
    int max_iterations = 1000;  // Maximum number of iterations

    // Perform the forward pass with training iterations and error convergence
    forward_pass(rbfnn, data, normalized_F, eta, lambda, tolerance, max_iterations);

    return 0;
}
