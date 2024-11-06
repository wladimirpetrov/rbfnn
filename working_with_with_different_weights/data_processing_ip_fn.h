#ifndef DATA_PROCESSING_IP_FN_H
#define DATA_PROCESSING_IP_FN_H

#include <vector>
#include <string>

// Struct to hold mission data
struct Mission {
    double m_value, M_value, L_value, g_value, d_value, b_value;
    double theta_cmd, theta_0;
    std::vector<double> F_values;
};

// Functions declared here, defined in data_processing_ip_fn.cpp
std::vector<Mission> read_csv(const std::string& filename);
void calculate_mean_std(const std::vector<Mission>& missions, Mission& mean, Mission& std_dev);
void calculate_normalized_values(const std::vector<Mission>& missions, const Mission& mean, const Mission& std_dev);

#endif // DATA_PROCESSING_IP_FN_H
