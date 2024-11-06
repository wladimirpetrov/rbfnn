#include "net_init.h"
#include "data_processing_output.h"  // Include the new header for output normalization
#include "training.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

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

// Compute activations for new inputs based on centers and widths
std::vector<double> compute_activations_for_new_input(const std::vector<double>& normalized_features, const RBFNN& rbfnn) {
    std::vector<double> activations(rbfnn.centers.size(), 0.0);

    for (size_t j = 0; j < rbfnn.centers.size(); ++j) {
        double distance_squared = 0.0;

        // Compute the Euclidean distance squared between normalized input and center c_j
        for (size_t d = 0; d < normalized_features.size(); ++d) {
            distance_squared += std::pow(normalized_features[d] - rbfnn.centers[j][d], 2);
        }

        // Compute Gaussian activation
        activations[j] = std::exp(-distance_squared / (2 * std::pow(rbfnn.widths[j], 2)));

        std::cout << "Activation of neuron " << j + 1 << ": " << activations[j] << std::endl;
    }

    return activations;
}

// Interpolation helper function
double interpolate_weight(double t_new, double t1, double t2, double w1, double w2) {
    return w1 + (t_new - t1) * (w2 - w1) / (t2 - t1);
}

std::vector<std::vector<double>> forward_pass(RBFNN& rbfnn, const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& F_values, double eta, double lambda, double tolerance, int max_iterations, const std::vector<double>& mean_F, const std::vector<double>& std_F) {
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

                        predicted_trajectories[i][t] += result;
                    }
                }
            }

            errors[i].resize(F_i.size(), 0.0);
            for (size_t t = 0; t < F_i.size(); ++t) {
                errors[i][t] = F_i[t] - predicted_trajectories[i][t];  // Element-wise difference
            }

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

                for (size_t j = 0; j < rbfnn.weights.size(); ++j) {
                    updated_weights[i][j][t] = rbfnn.weights[j][0] + weight_updates[j];  // Store updated weights
                }

                error_norm += calculate_norm(errors[i]);
            }
        }

        std::cout << "Total Error Norm: " << error_norm << std::endl;
        iteration++;
    }

    std::cout << "Training completed after " << iteration << " iterations." << std::endl;

    // Save the model (centers, widths, and weights)
    save_model(rbfnn, "trained_model.csv", updated_weights, F_values[0].size());

     // STEP 1: Introduce new theta_0 and theta_cmd
    double theta_0_new = 2.8;
    double theta_cmd_new = 0.0;
    std::cout << "\nNew Input Data: theta_0 = " << theta_0_new << ", theta_cmd = " << theta_cmd_new << std::endl;

    // STEP 2: Normalize new features using the mean and std dev from training
    double mean_theta_0 = 0;
    double std_theta_0 = 0.854;
    double mean_theta_cmd = 0.0;
    double std_theta_cmd = 0.0;

    double theta_0_new_norm = (theta_0_new - mean_theta_0) / std_theta_0;
    double theta_cmd_new_norm = (theta_cmd_new - mean_theta_cmd);

    std::vector<double> new_input_normalized = {theta_0_new_norm, theta_cmd_new_norm};
    std::cout << "Normalized theta_0_new: " << theta_0_new_norm << ", Normalized theta_cmd_new: " << theta_cmd_new_norm << std::endl;

    // STEP 3: Compute activations for the new normalized input
    std::vector<double> activations_new = compute_activations_for_new_input(new_input_normalized, rbfnn);

    // Introduce the trajectories for mission 1 and 3 for both neurons
    std::vector<double> mission1_neuron1 = {
    4.87645e-08, 4.87645e-08, 4.87645e-08, 4.87645e-08, 4.87645e-08, 4.87645e-08, 4.87645e-08, 4.87644e-08,
    4.87645e-08, 4.87645e-08, 4.87646e-08, 4.87644e-08, 4.87646e-08, 4.87644e-08, 4.87645e-08, 4.87646e-08,
    4.87646e-08, 4.87647e-08, 4.89394e-08, 4.87389e-08, 4.87161e-08, 4.86948e-08, 4.86506e-08, 4.84949e-08,
    -3.71765e-08, -4.89242e-08, -4.88608e-08, -4.88336e-08, -4.88187e-08, -4.88093e-08, -4.88027e-08,
    -4.87979e-08, -4.87941e-08, -4.87886e-08, -4.87865e-08, -4.87846e-08, -4.87831e-08, -4.87816e-08,
    -4.87803e-08, -4.87791e-08, -4.87781e-08, -4.8777e-08, -4.8776e-08, -4.87751e-08, -4.87742e-08,
    -4.87733e-08, -4.87725e-08, -4.87717e-08, -4.87709e-08, -4.87700e-08, -4.87692e-08, -4.87683e-08,
    -4.87675e-08, -4.87667e-08, -4.87656e-08, -4.87647e-08, -4.87636e-08, -4.87627e-08, -4.87615e-08,
    -4.87601e-08, -4.87588e-08, -4.8757e-08, -4.87554e-08, -4.87533e-08, -4.87508e-08, -4.87479e-08,
    -4.87444e-08, -4.87398e-08, -4.87338e-08, -4.87255e-08, -4.87130e-08, -4.86918e-08, -4.86486e-08,
    -4.85111e-08, -4.07875e-08, 4.89487e-08, 4.88743e-08, 4.88422e-08, 4.88251e-08, 4.88143e-08, 4.88068e-08,
    4.88014e-08, 4.87975e-08, 4.87941e-08, 4.87913e-08, 4.87889e-08, 4.87871e-08, 4.87852e-08, 4.87839e-08,
    4.87825e-08, 4.87812e-08, 4.87801e-08, 4.87789e-08, 4.87781e-08, 4.87771e-08, 4.87762e-08, 4.87755e-08,
    4.87743e-08, 4.87733e-08, 4.87727e-08, 4.8772e-08, 4.87709e-08, 4.87702e-08, 4.8769e-08, 4.8768e-08,
    4.87667e-08, 4.87657e-08, 4.87639e-08, 4.87625e-08, 4.87604e-08, 4.87577e-08, 4.8754e-08, 4.87485e-08,
    4.87392e-08, 4.87192e-08, 4.86388e-08, -4.89733e-08, -4.88333e-08, -4.88068e-08, -4.87955e-08, 
    -4.87892e-08, -4.87853e-08, -4.87824e-08, -4.87802e-08, -4.87786e-08, -4.87772e-08, -4.87759e-08,
    -4.87749e-08, -4.87739e-08, -4.87731e-08, -4.87723e-08, -4.87717e-08, -4.8771e-08, -4.87703e-08,
    -4.87698e-08, -4.87693e-08, -4.87687e-08, -4.87683e-08, -4.87677e-08, -4.87672e-08, -4.87668e-08,
    -4.87664e-08, -4.8766e-08, -4.87655e-08, -4.87651e-08, -4.87647e-08, -4.87644e-08, -4.87641e-08,
    -4.87637e-08, -4.87634e-08, -4.87631e-08, -4.87627e-08, -4.87624e-08, -4.87622e-08, -4.87619e-08,
    -4.87617e-08, -4.87615e-08, -4.87613e-08, -4.8761e-08, -4.87608e-08, -4.87607e-08, -4.87605e-08,
    -4.87603e-08, -4.87602e-08, -4.87599e-08, -4.87597e-08, -4.87598e-08, -4.87597e-08, -4.87596e-08,
    -4.87596e-08, -4.87595e-08, -4.87595e-08, -4.87595e-08, -4.87594e-08, -4.87595e-08, -4.87596e-08,
    -4.87596e-08, -4.87596e-08, -4.87596e-08, -4.87597e-08, -4.87598e-08, -4.87599e-08, -4.87599e-08,
    -4.876e-08, -4.876e-08, -4.87601e-08, -4.87602e-08, -4.87602e-08, -4.87602e-08, -4.87603e-08, 
    -4.87603e-08, -4.87604e-08};
    std::vector<double> mission1_neuron2 = {
    2.28762e-15, 2.28762e-15, 2.28762e-15, 2.28762e-15, 2.28762e-15, 2.28762e-15, 2.28762e-15, 2.28762e-15,
    2.28762e-15, 2.28762e-15, 2.28763e-15, 2.28762e-15, 2.28763e-15, 2.28762e-15, 2.28762e-15, 2.28763e-15,
    2.28763e-15, 2.28763e-15, 2.29583e-15, 2.28642e-15, 2.28535e-15, 2.28435e-15, 2.28228e-15, 2.27498e-15,
    -1.74401e-15, -2.29512e-15, -2.29214e-15, -2.29086e-15, -2.29017e-15, -2.28972e-15, -2.28941e-15,
    -2.28919e-15, -2.28901e-15, -2.28875e-15, -2.28865e-15, -2.28857e-15, -2.28849e-15, -2.28843e-15,
    -2.28836e-15, -2.28831e-15, -2.28826e-15, -2.28821e-15, -2.28816e-15, -2.28812e-15, -2.28808e-15,
    -2.28804e-15, -2.288e-15, -2.28796e-15, -2.28792e-15, -2.28788e-15, -2.28784e-15, -2.2878e-15,
    -2.28776e-15, -2.28772e-15, -2.28768e-15, -2.28763e-15, -2.28758e-15, -2.28754e-15, -2.28748e-15,
    -2.28742e-15, -2.28735e-15, -2.28727e-15, -2.28719e-15, -2.2871e-15, -2.28698e-15, -2.28684e-15,
    -2.28668e-15, -2.28646e-15, -2.28618e-15, -2.28579e-15, -2.28521e-15, -2.28421e-15, -2.28219e-15,
    -2.27574e-15, -1.91341e-15, 2.29626e-15, 2.29277e-15, 2.29127e-15, 2.29046e-15, 2.28996e-15, 2.28961e-15,
    2.28936e-15, 2.28917e-15, 2.28901e-15, 2.28888e-15, 2.28877e-15, 2.28868e-15, 2.2886e-15, 2.28853e-15,
    2.28847e-15, 2.2884e-15, 2.28835e-15, 2.2883e-15, 2.28826e-15, 2.28822e-15, 2.28817e-15, 2.28814e-15,
    2.28808e-15, 2.28804e-15, 2.28801e-15, 2.28797e-15, 2.28792e-15, 2.28789e-15, 2.28783e-15, 2.28779e-15,
    2.28772e-15, 2.28768e-15, 2.2876e-15, 2.28753e-15, 2.28743e-15, 2.2873e-15, 2.28713e-15, 2.28687e-15,
    2.28644e-15, 2.2855e-15, 2.28173e-15, -2.29742e-15, -2.29085e-15, -2.28961e-15, -2.28908e-15, 
    -2.28878e-15, -2.2886e-15, -2.28846e-15, -2.28836e-15, -2.28828e-15, -2.28822e-15, -2.28816e-15, 
    -2.28811e-15, -2.28806e-15, -2.28803e-15, -2.28799e-15, -2.28796e-15, -2.28793e-15, -2.2879e-15, 
    -2.28787e-15, -2.28785e-15, -2.28782e-15, -2.2878e-15, -2.28777e-15, -2.28775e-15, -2.28773e-15, 
    -2.28771e-15, -2.28769e-15, -2.28767e-15, -2.28765e-15, -2.28763e-15, -2.28762e-15, -2.2876e-15,
    -2.28759e-15, -2.28757e-15, -2.28756e-15, -2.28754e-15, -2.28753e-15, -2.28751e-15, -2.2875e-15, 
    -2.28749e-15, -2.28748e-15, -2.28747e-15, -2.28746e-15, -2.28745e-15, -2.28744e-15, -2.28743e-15, 
    -2.28742e-15, -2.28742e-15, -2.28741e-15, -2.2874e-15, -2.2874e-15, -2.2874e-15, -2.28739e-15, 
    -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.28739e-15, 
    -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.28739e-15, -2.2874e-15, -2.2874e-15, -2.28741e-15, 
    -2.28741e-15, -2.28741e-15, -2.28741e-15, -2.28742e-15, -2.28742e-15, -2.28742e-15, -2.28742e-15, 
    -2.28742e-15, -2.28743e-15, -2.28743e-15
};
    std::vector<double> mission3_neuron1 = {
    9.56261e-09, 9.56258e-09, 9.5625e-09, 9.56253e-09, 9.56254e-09, 9.56264e-09, 9.56243e-09, 9.56261e-09,
    9.56268e-09, 9.56267e-09, 9.5626e-09, 9.56271e-09, 9.56249e-09, 9.56273e-09, 9.56277e-09, 9.56239e-09,
    9.56247e-09, 9.56246e-09, 8.93499e-09, 9.62579e-09, 9.67935e-09, 9.72725e-09, 9.82049e-09, 1.01035e-08,
    1.85808e-09, -9.01697e-09, -9.27935e-09, -9.3687e-09, -9.41376e-09, -9.44138e-09, -9.46012e-09, 
    -9.47378e-09, -9.48419e-09, -9.49922e-09, -9.50497e-09, -9.50989e-09, -9.51414e-09, -9.51797e-09, 
    -9.5215e-09, -9.52447e-09, -9.52746e-09, -9.53014e-09, -9.53262e-09, -9.53513e-09, -9.53742e-09, 
    -9.53969e-09, -9.54178e-09, -9.54407e-09, -9.54615e-09, -9.54819e-09, -9.55038e-09, -9.5525e-09, 
    -9.55493e-09, -9.55694e-09, -9.55946e-09, -9.56197e-09, -9.56457e-09, -9.56727e-09, -9.57007e-09, 
    -9.57367e-09, -9.57687e-09, -9.58105e-09, -9.58552e-09, -9.59083e-09, -9.59691e-09, -9.60401e-09, 
    -9.61279e-09, -9.62368e-09, -9.63807e-09, -9.65773e-09, -9.68672e-09, -9.73349e-09, -9.82438e-09, 
    -1.00767e-08, -1.40035e-08, 8.87723e-09, 9.23169e-09, 9.34104e-09, 9.3948e-09, 9.42663e-09, 9.44826e-09, 
    9.46366e-09, 9.47518e-09, 9.48442e-09, 9.49187e-09, 9.49806e-09, 9.50319e-09, 9.50799e-09, 9.51199e-09, 
    9.51564e-09, 9.51891e-09, 9.52191e-09, 9.52479e-09, 9.52747e-09, 9.52987e-09, 9.53242e-09, 9.53465e-09, 
    9.5369e-09, 9.53937e-09, 9.54149e-09, 9.54363e-09, 9.546e-09, 9.54817e-09, 9.55101e-09, 9.55347e-09, 
    9.5565e-09, 9.55971e-09, 9.56352e-09, 9.56776e-09, 9.57311e-09, 9.57978e-09, 9.58899e-09, 9.60248e-09, 
    9.62527e-09, 9.67251e-09, 9.84412e-09, -8.66262e-09, -9.36905e-09, -9.44899e-09, -9.4805e-09, -9.4975e-09, 
    -9.5081e-09, -9.5159e-09, -9.52164e-09, -9.5261e-09, -9.52988e-09, -9.53303e-09, -9.53585e-09, -9.53829e-09, 
    -9.54039e-09, -9.54237e-09, -9.54422e-09, -9.54599e-09, -9.54759e-09, -9.54898e-09, -9.55047e-09, 
    -9.55178e-09, -9.55307e-09, -9.5543e-09, -9.55555e-09, -9.55659e-09, -9.55775e-09, -9.55891e-09, 
    -9.55983e-09, -9.56094e-09, -9.56182e-09, -9.56273e-09, -9.56364e-09, -9.56452e-09, -9.56544e-09, 
    -9.56623e-09, -9.56697e-09, -9.5677e-09, -9.56848e-09, -9.56909e-09, -9.56971e-09, -9.57036e-09, 
    -9.57092e-09, -9.57146e-09, -9.57191e-09, -9.5724e-09, -9.57278e-09, -9.57327e-09, -9.57356e-09, 
    -9.57409e-09, -9.57446e-09, -9.57454e-09, -9.57486e-09, -9.575e-09, -9.57498e-09, -9.57519e-09, 
    -9.57528e-09, -9.57525e-09, -9.5753e-09, -9.57518e-09, -9.57499e-09, -9.57498e-09, -9.5749e-09, 
    -9.57484e-09, -9.57469e-09, -9.57455e-09, -9.57438e-09, -9.5743e-09, -9.57407e-09, -9.5739e-09, 
    -9.57378e-09, -9.57372e-09, -9.57355e-09, -9.5734e-09, -9.57329e-09, -9.57325e-09, -9.57289e-09
};

    std::vector<double> mission3_neuron2 = {
    3.27214e-10, 3.27213e-10, 3.27211e-10, 3.27212e-10, 3.27212e-10, 3.27215e-10, 3.27208e-10, 3.27214e-10, 
    3.27217e-10, 3.27216e-10, 3.27214e-10, 3.27218e-10, 3.2721e-10, 3.27219e-10, 3.2722e-10, 3.27207e-10, 
    3.2721e-10, 3.27209e-10, 3.05739e-10, 3.29376e-10, 3.31209e-10, 3.32848e-10, 3.36039e-10, 3.45724e-10, 
    6.35799e-11, -3.08544e-10, -3.17522e-10, -3.20579e-10, -3.22121e-10, -3.23066e-10, -3.23707e-10, 
    -3.24175e-10, -3.24531e-10, -3.25046e-10, -3.25242e-10, -3.25411e-10, -3.25556e-10, -3.25687e-10, 
    -3.25808e-10, -3.2591e-10, -3.26012e-10, -3.26104e-10, -3.26188e-10, -3.26274e-10, -3.26353e-10, 
    -3.2643e-10, -3.26502e-10, -3.2658e-10, -3.26651e-10, -3.26721e-10, -3.26796e-10, -3.26869e-10, 
    -3.26952e-10, -3.2702e-10, -3.27107e-10, -3.27193e-10, -3.27282e-10, -3.27374e-10, -3.2747e-10, 
    -3.27593e-10, -3.27702e-10, -3.27846e-10, -3.27999e-10, -3.2818e-10, -3.28388e-10, -3.28631e-10, 
    -3.28932e-10, -3.29304e-10, -3.29797e-10, -3.30469e-10, -3.31461e-10, -3.33062e-10, -3.36172e-10, 
    -3.44805e-10, -4.79174e-10, 3.03762e-10, 3.15891e-10, 3.19633e-10, 3.21472e-10, 3.22562e-10, 
    3.23302e-10, 3.23829e-10, 3.24223e-10, 3.24539e-10, 3.24794e-10, 3.25006e-10, 3.25181e-10, 3.25346e-10, 
    3.25482e-10, 3.25607e-10, 3.25719e-10, 3.25822e-10, 3.25921e-10, 3.26012e-10, 3.26094e-10, 3.26181e-10, 
    3.26258e-10, 3.26335e-10, 3.26419e-10, 3.26492e-10, 3.26565e-10, 3.26646e-10, 3.2672e-10, 3.26817e-10, 
    3.26902e-10, 3.27006e-10, 3.27115e-10, 3.27246e-10, 3.27391e-10, 3.27574e-10, 3.27802e-10, 3.28117e-10, 
    3.28579e-10, 3.29359e-10, 3.30975e-10, 3.36847e-10, -2.96419e-10, -3.20591e-10, -3.23326e-10, 
    -3.24405e-10, -3.24987e-10, -3.25349e-10, -3.25616e-10, -3.25813e-10, -3.25965e-10, -3.26095e-10, 
    -3.26202e-10, -3.26299e-10, -3.26382e-10, -3.26454e-10, -3.26522e-10, -3.26585e-10, -3.26646e-10, 
    -3.267e-10, -3.26748e-10, -3.26799e-10, -3.26844e-10, -3.26888e-10, -3.2693e-10, -3.26973e-10, 
    -3.27009e-10, -3.27048e-10, -3.27088e-10, -3.27119e-10, -3.27157e-10, -3.27188e-10, -3.27219e-10, 
    -3.2725e-10, -3.2728e-10, -3.27311e-10, -3.27338e-10, -3.27364e-10, -3.27389e-10, -3.27415e-10, 
    -3.27436e-10, -3.27457e-10, -3.2748e-10, -3.27499e-10, -3.27517e-10, -3.27533e-10, -3.2755e-10, 
    -3.27563e-10, -3.27579e-10, -3.27589e-10, -3.27607e-10, -3.2762e-10, -3.27623e-10, -3.27634e-10, 
    -3.27639e-10, -3.27638e-10, -3.27645e-10, -3.27648e-10, -3.27647e-10, -3.27649e-10, -3.27645e-10, 
    -3.27638e-10, -3.27638e-10, -3.27635e-10, -3.27633e-10, -3.27628e-10, -3.27623e-10, -3.27617e-10, 
    -3.27615e-10, -3.27607e-10, -3.27601e-10, -3.27597e-10, -3.27595e-10, -3.27589e-10, -3.27584e-10, 
    -3.2758e-10, -3.27579e-10, -3.27566e-10
};

    // Set the time range based on the missions (t1 for mission 1 and t2 for mission 3)
    double t1 = 3.0;  // mission 1's theta_0 = 3
    double t2 = 2.0;  // mission 3's theta_0 = 2
    double t_new = theta_0_new;  // The new theta_0 = 2.8

    // Interpolate weights for neuron 1 and neuron 2 at each time step
    std::vector<double> interpolated_weights_neuron1;
    std::vector<double> interpolated_weights_neuron2;

    for (size_t t = 0; t < mission1_neuron1.size(); ++t) {
        double interpolated_weight_neuron1 = interpolate_weight(t_new, t1, t2, mission1_neuron1[t], mission3_neuron1[t]);
        double interpolated_weight_neuron2 = interpolate_weight(t_new, t1, t2, mission1_neuron2[t], mission3_neuron2[t]);

        interpolated_weights_neuron1.push_back(interpolated_weight_neuron1);
        interpolated_weights_neuron2.push_back(interpolated_weight_neuron2);
        std::cout << "Time step " << t << ": " << std::endl;
        std::cout << "Interpolated weight for Neuron 1: " << interpolated_weight_neuron1 << std::endl;
        std::cout << "Interpolated weight for Neuron 2: " << interpolated_weight_neuron2 << std::endl;
    }

    // STEP 5: Compute the predicted control trajectory using the interpolated weights and activations
    std::vector<double> predicted_trajectory(interpolated_weights_neuron1.size());

    for (size_t t = 0; t < interpolated_weights_neuron1.size(); ++t) {
        predicted_trajectory[t] = activations_new[0] * interpolated_weights_neuron1[t] + activations_new[1] * interpolated_weights_neuron2[t];
    }

    // Output the predicted trajectory (in normalized form, not yet denormalized)
    std::cout << "Predicted normalized trajectory (multiple time steps):\n";
    for (size_t t = 0; t < predicted_trajectory.size(); ++t) {
        std::cout << "t = " << t << ": " << predicted_trajectory[t] << std::endl;
    }

    // If you'd prefer to output the trajectory as a single line vector instead of each time step individually:
    std::cout << "Predicted normalized trajectory vector:\n";
    std::cout << "[";
    for (size_t t = 0; t < predicted_trajectory.size(); ++t) {
        std::cout << predicted_trajectory[t];
        if (t != predicted_trajectory.size() - 1) {
            std::cout << ", ";  // Add a comma between values
        }
    }
    std::cout << "]" << std::endl;

        // STEP 6: Denormalize the predicted trajectory
    std::vector<double> denormalized_trajectory(predicted_trajectory.size());

    for (size_t t = 0; t < predicted_trajectory.size(); ++t) {
        denormalized_trajectory[t] = predicted_trajectory[t] * std_F[t] + mean_F[t];  // Denormalize using mean and std dev
    }

    // Output the denormalized trajectory
    std::cout << "Denormalized trajectory (multiple time steps):\n";
    for (size_t t = 0; t < denormalized_trajectory.size(); ++t) {
        std::cout << "t = " << t << ": " << denormalized_trajectory[t] << std::endl;
    }

    // If you'd prefer to output the trajectory as a single line vector instead of each time step individually:
    std::cout << "Denormalized trajectory vector:\n";
    std::cout << "[";
    for (size_t t = 0; t < denormalized_trajectory.size(); ++t) {
        std::cout << denormalized_trajectory[t];
        if (t != denormalized_trajectory.size() - 1) {
            std::cout << ", ";  // Add a comma between values
        }
    }
    std::cout << "]" << std::endl;

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
    forward_pass(rbfnn, data, normalized_F, eta, lambda, tolerance, max_iterations, mean_F, std_F);

    return 0;
}
