#include "net_init.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>  // For random number generation

// Helper function to compute Euclidean distance between two points
double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(distance);
}

// Helper function to perform K-means clustering
std::vector<int> k_means_clustering(const std::vector<std::vector<double>>& data, int r, std::vector<std::vector<double>>& centers) {
    int n = data.size();
    int input_dim = data[0].size();
    std::vector<int> assignments(n, 0);
    std::vector<int> counts(r, 0);

    // Initialize centers randomly
    for (int j = 0; j < r; ++j) {
        centers[j] = data[std::rand() % n];
    }

    bool changed = true;
    while (changed) {
        changed = false;

        // Step 1: Assign data points to closest center
        for (int i = 0; i < n; ++i) {
            double min_distance = euclidean_distance(data[i], centers[0]);
            int closest_center = 0;
            for (int j = 1; j < r; ++j) {
                double distance = euclidean_distance(data[i], centers[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_center = j;
                }
            }

            if (assignments[i] != closest_center) {
                changed = true;
                assignments[i] = closest_center;
            }
        }

        // Step 2: Update centers
        centers.assign(r, std::vector<double>(input_dim, 0.0));
        counts.assign(r, 0);
        for (int i = 0; i < n; ++i) {
            int cluster = assignments[i];
            for (int j = 0; j < input_dim; ++j) {
                centers[cluster][j] += data[i][j];
            }
            counts[cluster]++;
        }

        for (int j = 0; j < r; ++j) {
            if (counts[j] > 0) {
                for (int k = 0; k < input_dim; ++k) {
                    centers[j][k] /= counts[j];
                }
            }
        }
    }

    return assignments;
}

// Function to initialize the RBFNN parameters
RBFNN initialize_rbfnn(int r, int input_dim, const std::vector<std::vector<double>>& data) {
    RBFNN rbfnn;
    rbfnn.centers.resize(r, std::vector<double>(input_dim, 0.0));
    rbfnn.widths.resize(r, 0.0);
    rbfnn.weights.resize(r, std::vector<double>(1, 0.0));  // Initialize weights to zero (output size 1 for simplicity)

    // Step 1: Perform K-means clustering to find centers
    std::vector<int> assignments = k_means_clustering(data, r, rbfnn.centers);

    // Step 2: Calculate widths (average distance within each cluster)
    for (int j = 0; j < r; ++j) {
        double total_distance = 0.0;
        int count = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            if (assignments[i] == j) {
                total_distance += euclidean_distance(data[i], rbfnn.centers[j]);
                count++;
            }
        }
        rbfnn.widths[j] = (count > 0) ? total_distance / count : 1.0;  // Avoid division by zero
    }

    // Step 3: Initialize weights to zero
    for (int j = 0; j < r; ++j) {
        rbfnn.weights[j].assign(1, 0.0);  // For now we are initializing 1 weight per center, change if needed
    }

    return rbfnn;
}
