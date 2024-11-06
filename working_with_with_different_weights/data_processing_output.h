#ifndef DATA_PROCESSING_OUTPUT_H
#define DATA_PROCESSING_OUTPUT_H

#include "data_processing_ip_fn.h"  // Reuse the same structure

#include <vector>

// Function to calculate mean and standard deviation for F_values
void calculate_mean_std_F(const std::vector<Mission>& missions, std::vector<double>& mean_F, std::vector<double>& std_F);

// Function to normalize F trajectories
std::vector<std::vector<double>> normalize_F_trajectories(const std::vector<Mission>& missions, const std::vector<double>& mean_F, const std::vector<double>& std_F);

#endif  // DATA_PROCESSING_OUTPUT_H

