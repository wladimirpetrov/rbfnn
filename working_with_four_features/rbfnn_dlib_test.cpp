#include <iostream>
#include "../../rbfnn_lib/dlib/dlib/svm.h>"
#include "data_gather.h"

// Typedef for convenience
typedef dlib::matrix<double> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;

int main() {
    // Gather data from the CSV file
    std::vector<Mission> missions = read_csv("data.csv");

    // Create samples (input features) and labels (output trajectories)
    std::vector<sample_type> samples;
    std::vector<double> labels;

    for (const auto& mission : missions) {
        sample_type sample(2, 1);  // Two features: theta_0 and theta_cmd
        sample(0, 0) = mission.theta_0;
        sample(1, 0) = mission.theta_cmd;

        samples.push_back(sample);

        // Assume you want to predict the first element of the F trajectory for simplicity
        if (!mission.F_values.empty()) {
            labels.push_back(mission.F_values[0]);
        } else {
            labels.push_back(0.0);  // Default to 0 if no trajectory is found
        }
    }

    // Create the Kernel Ridge Regression trainer
    dlib::krr_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(0.1));  // Set RBF kernel with gamma = 0.1

    // Train the model
    dlib::decision_function<kernel_type> df = trainer.train(samples, labels);

    // Predict the F trajectory for a new mission
    sample_type test_sample(2, 1);
    test_sample(0, 0) = 1.2;  // Example theta_0
    test_sample(1, 0) = 0.0;  // Example theta_cmd

    double predicted_f = df(test_sample);
    std::cout << "Predicted F value: " << predicted_f << std::endl;

    return 0;
}
