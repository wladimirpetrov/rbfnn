#include "data_processing_output.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

// Function to calculate mean and standard deviation for F_values (trajectories)
void calculate_mean_std_F(const std::vector<Mission>& missions, std::vector<double>& mean_F, std::vector<double>& std_F) {
    int num_missions = missions.size();
    int num_steps = missions[0].F_values.size();  // Assuming all missions have the same number of time steps

    mean_F.resize(num_steps, 0.0);
    std_F.resize(num_steps, 0.0);

    // Calculate mean for each time step
    for (const auto& mission : missions) {
        for (size_t t = 0; t < mission.F_values.size(); ++t) {
            mean_F[t] += mission.F_values[t];
        }
    }
    for (size_t t = 0; t < mean_F.size(); ++t) {
        mean_F[t] /= num_missions;
    }

    // Calculate standard deviation for each time step
    for (const auto& mission : missions) {
        for (size_t t = 0; t < mission.F_values.size(); ++t) {
            std_F[t] += std::pow(mission.F_values[t] - mean_F[t], 2);
        }
    }
    for (size_t t = 0; t < std_F.size(); ++t) {
        std_F[t] = std::sqrt(std_F[t] / num_missions);
    }
}

// Function to normalize F trajectories
std::vector<std::vector<double>> normalize_F_trajectories(const std::vector<Mission>& missions, const std::vector<double>& mean_F, const std::vector<double>& std_F) {
    std::vector<std::vector<double>> normalized_F(missions.size());

    for (size_t i = 0; i < missions.size(); ++i) {
        normalized_F[i].resize(missions[i].F_values.size());
        for (size_t t = 0; t < missions[i].F_values.size(); ++t) {
            if (std_F[t] != 0) {
                normalized_F[i][t] = (missions[i].F_values[t] - mean_F[t]) / std_F[t];
            } else {
                normalized_F[i][t] = 0.0;  // Avoid division by zero
            }
        }
    }

    return normalized_F;
}

