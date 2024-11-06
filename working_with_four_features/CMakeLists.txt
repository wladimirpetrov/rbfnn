cmake_minimum_required(VERSION 3.5)
project(klqr_ws)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR} # Include current source directory
)

# Add executables
add_executable(klqr_node src/klqr_node.cpp)
add_executable(printing_node src/printing_node.cpp)

ament_target_dependencies(klqr_node rclcpp std_msgs)
ament_target_dependencies(printing_node rclcpp std_msgs)

# Install targets
install(TARGETS
  klqr_node
  printing_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()

