#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"

class KLRQNode : public rclcpp::Node {
public:
    KLRQNode() : Node("klqr_node") {
        
        x_pos_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/x_pos", 10, std::bind(&KLRQNode::xPosCallback, this, std::placeholders::_1));

        xdot_acc_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/xdot_acc", 10, std::bind(&KLRQNode::xdotAccCallback, this, std::placeholders::_1));

        theta_ang_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/theta_ang", 10, std::bind(&KLRQNode::thetaAngCallback, this, std::placeholders::_1));

        thetadot_rate_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/thetadot_rate", 10, std::bind(&KLRQNode::thetadotRateCallback, this, std::placeholders::_1));

        force_pub_ = this->create_publisher<std_msgs::msg::Float64>("/force", 10);

        x_pos_ = 0.0;
        xdot_acc_ = 0.0;
        theta_ang_ = 0.0;
        thetadot_rate_ = 0.0;
        
        K_ = {-1.00001469341606e-11, -2.00001469310717, 20029.1047244970, 8952.81065268808}; 
    }

private:

    void xPosCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        x_pos_ = msg->data;
    }

    void xdotAccCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        xdot_acc_ = msg->data;
    }

    void thetaAngCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        theta_ang_ = msg->data;
    }

    void thetadotRateCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        thetadot_rate_ = msg->data;
        computeForce();  
    }

    void computeForce() {
        
        std::vector<double> u = {x_pos_, xdot_acc_, theta_ang_, thetadot_rate_};

        double force = 0.0;
        for (size_t i = 0; i < K_.size(); ++i) {
            force -= K_[i] * u[i];  
        }

        auto force_msg = std_msgs::msg::Float64();
        force_msg.data = force;
        force_pub_->publish(force_msg);

        RCLCPP_INFO(this->get_logger(), "Computed force: %f", force);
    }

    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr x_pos_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr xdot_acc_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr theta_ang_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr thetadot_rate_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr force_pub_;

    double x_pos_;
    double xdot_acc_;
    double theta_ang_;
    double thetadot_rate_;

    std::vector<double> K_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KLRQNode>());
    rclcpp::shutdown();
    return 0;
}

//vpetrov@lnx-me002:~/dev/inv_pend_test/klqr_ws$ colcon build
//vpetrov@lnx-me002:~/dev/inv_pend_test/klqr_ws$ source install/setup.bash
//vpetrov@lnx-me002:~/dev/inv_pend_test/klqr_ws$ ros2 run klqr_ws klqr_node
//vpetrov@lnx-me002:~/dev/inv_pend_test/klqr_ws$ rm -rf build install log


