#include "data_processing_ip_fn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Function to safely convert a string to a double (returns 0 if the string is invalid)
double safe_stod(const std::string& str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& e) {
        return 0.0;
    } catch (const std::out_of_range& e) {
        return 0.0;
    }
}

// Function to read CSV file and process data
std::vector<Mission> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Mission> missions;
    std::string line;

    while (std::getline(file, line)) {
        if (line.find("m_value") != std::string::npos) {
            continue;
        }

        Mission mission;
        std::stringstream ss(line);
        std::string token;

        // extracting values for m_value, M_value, etc.
        std::getline(ss, token, ','); mission.m_value = safe_stod(token);
        std::getline(ss, token, ','); mission.M_value = safe_stod(token);
        std::getline(ss, token, ','); mission.L_value = safe_stod(token);
        std::getline(ss, token, ','); mission.g_value = safe_stod(token);
        std::getline(ss, token, ','); mission.d_value = safe_stod(token);
        std::getline(ss, token, ','); mission.b_value = safe_stod(token);
        std::getline(ss, token, ','); mission.theta_cmd = safe_stod(token);
        std::getline(ss, token, ','); mission.theta_0 = safe_stod(token);

        // read the F values
        mission.F_values.clear();
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            for (int i = 0; i < 8; ++i) {
                std::getline(ss, token, ',');
            }
            std::getline(ss, token, ',');
            if (line.find("m_value") != std::string::npos) {
                break;
            }
            if (!token.empty()) {
                double F_value = safe_stod(token);
                mission.F_values.push_back(F_value);
            }
        }
        missions.push_back(mission);
    }
    return missions;
}

// f-n to calculate mean and standard deviation for each parameter
void calculate_mean_std(const std::vector<Mission>& missions, Mission& mean, Mission& std_dev) {
    int n = missions.size();
    mean = {0, 0, 0, 0, 0, 0, 0, 0};
    std_dev = {0, 0, 0, 0, 0, 0, 0, 0};

    // Calculate mean
    for (const auto& mission : missions) {
        mean.m_value += mission.m_value;
        mean.M_value += mission.M_value;
        mean.L_value += mission.L_value;
        mean.g_value += mission.g_value;
        mean.d_value += mission.d_value;
        mean.b_value += mission.b_value;
        mean.theta_cmd += mission.theta_cmd;
        mean.theta_0 += mission.theta_0;
    }

    mean.m_value /= n;
    mean.M_value /= n;
    mean.L_value /= n;
    mean.g_value /= n;
    mean.d_value /= n;
    mean.b_value /= n;
    mean.theta_cmd /= n;
    mean.theta_0 /= n;

    // Calculate standard deviation
    for (const auto& mission : missions) {
        std_dev.m_value += std::pow(mission.m_value - mean.m_value, 2);
        std_dev.M_value += std::pow(mission.M_value - mean.M_value, 2);
        std_dev.L_value += std::pow(mission.L_value - mean.L_value, 2);
        std_dev.g_value += std::pow(mission.g_value - mean.g_value, 2);
        std_dev.d_value += std::pow(mission.d_value - mean.d_value, 2);
        std_dev.b_value += std::pow(mission.b_value - mean.b_value, 2);
        std_dev.theta_cmd += std::pow(mission.theta_cmd - mean.theta_cmd, 2);
        std_dev.theta_0 += std::pow(mission.theta_0 - mean.theta_0, 2);
    }

    std_dev.m_value = std::sqrt(std_dev.m_value / n);
    std_dev.M_value = std::sqrt(std_dev.M_value / n);
    std_dev.L_value = std::sqrt(std_dev.L_value / n);
    std_dev.g_value = std::sqrt(std_dev.g_value / n);
    std_dev.d_value = std::sqrt(std_dev.d_value / n);
    std_dev.b_value = std::sqrt(std_dev.b_value / n);
    std_dev.theta_cmd = std::sqrt(std_dev.theta_cmd / n);
    std_dev.theta_0 = std::sqrt(std_dev.theta_0 / n);
}

// Function to compute normalized values xi' for each mission and parameter
void calculate_normalized_values(const std::vector<Mission>& missions, const Mission& mean, const Mission& std_dev) {
    auto safe_normalize = [](double value, double mean, double std_dev) {
        return std_dev == 0.0 ? 0.0 : (value - mean) / std_dev;
    };
    
    for (size_t i = 0; i < missions.size(); ++i) {
        const auto& mission = missions[i];
        std::cout << "Normalized values for Mission " << i + 1 << ": " << std::endl;
        std::cout << "m_value': " << safe_normalize(mission.m_value, mean.m_value, std_dev.m_value) << std::endl;
        std::cout << "M_value': " << safe_normalize(mission.M_value, mean.M_value, std_dev.M_value) << std::endl;
        std::cout << "L_value': " << safe_normalize(mission.L_value, mean.L_value, std_dev.L_value) << std::endl;
        std::cout << "g_value': " << safe_normalize(mission.g_value, mean.g_value, std_dev.g_value) << std::endl;
        std::cout << "d_value': " << safe_normalize(mission.d_value, mean.d_value, std_dev.d_value) << std::endl;
        std::cout << "b_value': " << safe_normalize(mission.b_value, mean.b_value, std_dev.b_value) << std::endl;
        std::cout << "theta_cmd': " << safe_normalize(mission.theta_cmd, mean.theta_cmd, std_dev.theta_cmd) << std::endl;
        std::cout << "theta_0': " << safe_normalize(mission.theta_0, mean.theta_0, std_dev.theta_0) << std::endl;
    }
}
