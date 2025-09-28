#!/bin/bash

# LM Studio Test Runner Script

echo "LM Studio Test Suite"
echo "===================="

# Check if ROS2 is available
if ! command -v ros2 &> /dev/null; then
    echo "Error: ROS2 not found. Please source your ROS2 installation."
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  client     - Run client tests only"
    echo "  ros        - Run ROS node tests only"
    echo "  integration - Run integration test (requires LM Studio running)"
    echo "  all        - Run all tests (default)"
}

# Function to run client tests
run_client_tests() {
    echo "Running client tests..."
    python3 test_lm_studio_client.py
}

# Function to run ROS node tests
run_ros_tests() {
    echo "Running ROS node tests..."
    python3 test_lm_studio_ros_node.py
}

# Function to run integration tests
run_integration_tests() {
    echo "Running integration tests..."
    echo "Note: Ensure LM Studio is running on localhost:1234"
    python3 test_integration.py
}

# Parse command line arguments
case "${1:-all}" in
    "client")
        run_client_tests
        ;;
    "ros")
        run_ros_tests
        ;;
    "integration")
        run_integration_tests
        ;;
    "all")
        run_client_tests
        echo
        run_ros_tests
        echo
        run_integration_tests
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo "Test suite completed."
