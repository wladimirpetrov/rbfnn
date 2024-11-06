#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>  // For std::numeric_limits

// Struct to hold mission data
struct Mission {
    double m_value, M_value, L_value, g_value, d_value, b_value;
    double theta_cmd, theta_0;
    std::vector<double> F_values;
};

// Function to safely convert a string to a double (returns 0 if the string is invalid)
double safe_stod(const std::string& str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& e) {
        return 0.0;  // Default value if conversion fails
    } catch (const std::out_of_range& e) {
        return 0.0;  // Default value if out of range
    }
}

// Function to read CSV file and process data
std::vector<Mission> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Mission> missions;
    std::string line;

    int mission_number = 0;  // Track mission number
    while (std::getline(file, line)) {
        if (line.find("m_value") != std::string::npos) {  // Check if it's a header line
            // Skip the header line
            continue;
        }

        mission_number++;  // Increment mission number for each mission

        Mission mission;
        std::stringstream ss(line);
        std::string token;

        // Extract values for m_value, M_value, etc., with safe conversions
        std::getline(ss, token, ','); mission.m_value = safe_stod(token);
        std::getline(ss, token, ','); mission.M_value = safe_stod(token);
        std::getline(ss, token, ','); mission.L_value = safe_stod(token);
        std::getline(ss, token, ','); mission.g_value = safe_stod(token);
        std::getline(ss, token, ','); mission.d_value = safe_stod(token);
        std::getline(ss, token, ','); mission.b_value = safe_stod(token);
        std::getline(ss, token, ','); mission.theta_cmd = safe_stod(token);
        std::getline(ss, token, ','); mission.theta_0 = safe_stod(token);

        // Now read the trajectory F values (9th column / I column)
        mission.F_values.clear();
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            
            // We need to go directly to the 9th column to find the F_value
            for (int i = 0; i < 8; ++i) {
                std::getline(ss, token, ',');  // Skip the first 8 columns
            }
            std::getline(ss, token, ',');  // Now we get the F_value from the 9th column

            // Check if the line is part of the next mission's header (which contains 'm_value')
            if (line.find("m_value") != std::string::npos) {
                break;  // Found the next mission's header
            }

            // Check if token is non-empty, and if so, add it to F_values
            if (!token.empty()) {
                double F_value = safe_stod(token);  // Extract F_value
                mission.F_values.push_back(F_value);  // Add F_value to vector
            }
        }

        // Print mission number and extracted values for verification
        std::cout << "Mission " << mission_number << ": "
                  << "m_value=" << mission.m_value << ", "
                  << "M_value=" << mission.M_value << ", "
                  << "L_value=" << mission.L_value << ", "
                  << "g_value=" << mission.g_value << ", "
                  << "d_value=" << mission.d_value << ", "
                  << "b_value=" << mission.b_value << ", "
                  << "theta_cmd=" << mission.theta_cmd << ", "
                  << "theta_0=" << mission.theta_0 << std::endl;

        // Print the extracted force values (F_values)
        std::cout << "Force values (F_values): ";
        for (const auto& F : mission.F_values) {
            std::cout << F << " ";  // Print each F_value
        }
        std::cout << std::endl;

        missions.push_back(mission);  // Add mission to the list
    }

    return missions;
}

// Function to calculate mean and standard deviation for each parameter
void calculate_mean_std(const std::vector<Mission>& missions, Mission& mean, Mission& std_dev) {
    int n = missions.size();

    // Initialize mean and std_dev to 0
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
    for (size_t i = 0; i < missions.size(); ++i) {
        const auto& mission = missions[i];
        std::cout << "Normalized values for Mission " << i + 1 << ": " << std::endl;

        // Check if standard deviation is zero, assign 0 to normalized values in that case
        auto safe_normalize = [](double value, double mean, double std_dev) {
            return std_dev == 0.0 ? 0.0 : (value - mean) / std_dev;
        };
        
        // Compute normalized values for each parameter using safe_normalize
        double m_value_norm = safe_normalize(mission.m_value, mean.m_value, std_dev.m_value);
        double M_value_norm = safe_normalize(mission.M_value, mean.M_value, std_dev.M_value);
        double L_value_norm = safe_normalize(mission.L_value, mean.L_value, std_dev.L_value);
        double g_value_norm = safe_normalize(mission.g_value, mean.g_value, std_dev.g_value);
        double d_value_norm = safe_normalize(mission.d_value, mean.d_value, std_dev.d_value);
        double b_value_norm = safe_normalize(mission.b_value, mean.b_value, std_dev.b_value);
        double theta_cmd_norm = safe_normalize(mission.theta_cmd, mean.theta_cmd, std_dev.theta_cmd);
        double theta_0_norm = safe_normalize(mission.theta_0, mean.theta_0, std_dev.theta_0);

        // Output normalized values
        std::cout << "m_value': " << m_value_norm << std::endl;
        std::cout << "M_value': " << M_value_norm << std::endl;
        std::cout << "L_value': " << L_value_norm << std::endl;
        std::cout << "g_value': " << g_value_norm << std::endl;
        std::cout << "d_value': " << d_value_norm << std::endl;
        std::cout << "b_value': " << b_value_norm << std::endl;
        std::cout << "theta_cmd': " << theta_cmd_norm << std::endl;
        std::cout << "theta_0': " << theta_0_norm << std::endl;
    }
}

// Main function
int main() {
    std::string filename = "data.csv";
    std::vector<Mission> missions = read_csv(filename);

    Mission mean, std_dev;
    calculate_mean_std(missions, mean, std_dev);

    // Output mean and standard deviation
    std::cout << "Mean Values: "
              << "m_value=" << mean.m_value << ", "
              << "M_value=" << mean.M_value << ", "
              << "L_value=" << mean.L_value << ", "
              << "g_value=" << mean.g_value << ", "
              << "d_value=" << mean.d_value << ", "
              << "b_value=" << mean.b_value << ", "
              << "theta_cmd=" << mean.theta_cmd << ", "
              << "theta_0=" << mean.theta_0 << std::endl;

    std::cout << "Standard Deviation: "
              << "m_value=" << std_dev.m_value << ", "
              << "M_value=" << std_dev.M_value << ", "
              << "L_value=" << std_dev.L_value << ", "
              << "g_value=" << std_dev.g_value << ", "
              << "d_value=" << std_dev.d_value << ", "
              << "b_value=" << std_dev.b_value << ", "
              << "theta_cmd=" << std_dev.theta_cmd << ", "
              << "theta_0=" << std_dev.theta_0 << std::endl;

    // Compute and output normalized values
    calculate_normalized_values(missions, mean, std_dev);

    return 0;
}
