#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

class PrintingNode : public rclcpp::Node {
public:
    PrintingNode()
        : Node("printing_node"), static_values_saved_(false), last_saved_t0_(-0.05) {  // Start with t0 below 0 to trigger the first save.

        output_file_.open("data.csv", std::ios::out | std::ios::app);
        if (!output_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open data.csv file for writing.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Successfully opened data.csv.");
            // headers
            output_file_ << "m_value,M_value,L_value,g_value,d_value,b_value,theta_cmd,theta_0,F_value,t0\n";  // CSV Headers
            output_file_.flush();  
        }

        received_m_ = received_M_ = received_L_ = received_g_ = received_d_ = received_b_ = received_theta_cmd_ = received_theta_act_ = false;

        m_sub_ = this->create_subscription<std_msgs::msg::Float64>("/m_value", 10, std::bind(&PrintingNode::mCallback, this, std::placeholders::_1));
        M_sub_ = this->create_subscription<std_msgs::msg::Float64>("/M_value", 10, std::bind(&PrintingNode::MCallback, this, std::placeholders::_1));
        L_sub_ = this->create_subscription<std_msgs::msg::Float64>("/L_value", 10, std::bind(&PrintingNode::LCallback, this, std::placeholders::_1));
        g_sub_ = this->create_subscription<std_msgs::msg::Float64>("/g_value", 10, std::bind(&PrintingNode::gCallback, this, std::placeholders::_1));
        d_sub_ = this->create_subscription<std_msgs::msg::Float64>("/d_value", 10, std::bind(&PrintingNode::dCallback, this, std::placeholders::_1));
        b_sub_ = this->create_subscription<std_msgs::msg::Float64>("/b_value", 10, std::bind(&PrintingNode::bCallback, this, std::placeholders::_1));
        theta_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>("/theta_cmd", 10, std::bind(&PrintingNode::thetaCmdCallback, this, std::placeholders::_1));
        theta_act_sub_ = this->create_subscription<std_msgs::msg::Float64>("/theta_ang", 10, std::bind(&PrintingNode::thetaActCallback, this, std::placeholders::_1));

        F_sub_ = this->create_subscription<std_msgs::msg::Float64>("/F_value", 10, std::bind(&PrintingNode::FCallback, this, std::placeholders::_1));
        t0_sub_ = this->create_subscription<std_msgs::msg::Float64>("/t0", 10, std::bind(&PrintingNode::t0Callback, this, std::placeholders::_1));
    }

private:
    
    void mCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        m_value_ = msg->data; 
        received_m_ = true; 
        checkAndSaveStaticValues(); 
    }
    void MCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        M_value_ = msg->data; 
        received_M_ = true; 
        checkAndSaveStaticValues(); 
    }
    void LCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        L_value_ = msg->data; 
        received_L_ = true; 
        checkAndSaveStaticValues(); 
    }
    void gCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        g_value_ = msg->data; 
        received_g_ = true; 
        checkAndSaveStaticValues(); 
    }
    void dCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        d_value_ = msg->data; 
        received_d_ = true; 
        checkAndSaveStaticValues(); 
    }
    void bCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        b_value_ = msg->data; 
        received_b_ = true; 
        checkAndSaveStaticValues(); 
    }
    void thetaCmdCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        theta_cmd_ = msg->data; 
        received_theta_cmd_ = true; 
        checkAndSaveStaticValues(); 
    }
    void thetaActCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        theta_act_ = msg->data; 
        received_theta_act_ = true; 
        checkAndSaveStaticValues(); 
    }

    void checkAndSaveStaticValues() {
        if (!static_values_saved_ && received_m_ && received_M_ && received_L_ && received_g_ && received_d_ && received_b_ && received_theta_cmd_ && received_theta_act_) {
            saveStaticValues();
            static_values_saved_ = true; 
        }
    }

    void saveStaticValues() {
        output_file_ << m_value_ << "," << M_value_ << "," << L_value_ << "," << g_value_ << ","
                     << d_value_ << "," << b_value_ << "," << theta_cmd_ << "," << theta_act_ << ",,\n";
        output_file_.flush(); 
    }

    // Callback functions for dynamic values (saved on every call after static values are saved)
    void FCallback(const std_msgs::msg::Float64::SharedPtr msg) { 
        F_value_ = msg->data; 
    }

    void t0Callback(const std_msgs::msg::Float64::SharedPtr msg) { 
        t0_ = msg->data; 
        if (static_values_saved_ && shouldSaveDynamicValues()) {
            saveDynamicValues(); 
        }
    }

    bool shouldSaveDynamicValues() {
        double interval = 0.05;
        double epsilon = 0.001;  // Tolerance for floating-point comparison
        return (std::abs(t0_ - (last_saved_t0_ + interval)) < epsilon) || t0_ == 0.05;
    }

    void saveDynamicValues() {
        output_file_ << ",,,,,,,," << F_value_ << "," << t0_ << "\n";  
        output_file_.flush();
        last_saved_t0_ = t0_;  // Update last saved t0
    }

    double m_value_, M_value_, L_value_, g_value_, d_value_, b_value_, theta_cmd_, theta_act_;
    double F_value_, t0_;

    bool received_m_, received_M_, received_L_, received_g_, received_d_, received_b_, received_theta_cmd_, received_theta_act_;

    bool static_values_saved_;
    double last_saved_t0_;  // Store the last saved t0 value

    std::ofstream output_file_;

    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr m_sub_, M_sub_, L_sub_, g_sub_, d_sub_, b_sub_, theta_cmd_sub_, theta_act_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr F_sub_, t0_sub_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PrintingNode>());
    rclcpp::shutdown();
    return 0;
}

//vpetrov@lnx-me002:/local/vol00/home/vpetrov/dev/inv_pend_test/klqr_ws$ source install/setup.bash
// vpetrov@lnx-me002:/local/vol00/home/vpetrov/dev/inv_pend_test/klqr_ws$ ros2 run klqr_ws printing_node 
// 
