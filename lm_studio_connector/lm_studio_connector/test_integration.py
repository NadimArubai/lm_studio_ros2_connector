#!/usr/bin/env python3

import subprocess
import time
import sys
import os
import signal

def start_lm_studio_node():
    """Start the LM Studio ROS2 node"""
    print("Starting LM Studio ROS2 node...")
    
    # This assumes your node is installed or available in the current environment
    process = subprocess.Popen(
        ['ros2', 'run', 'lm_studio_connector', 'lm_studio_node'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Important for proper signal handling
    )
    
    time.sleep(5)  # Give node time to start
    return process

def stop_node(process):
    """Stop the node process"""
    print("Stopping LM Studio node...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait()

def run_integration_test():
    """Run integration test with actual node"""
    print("=== LM Studio Integration Test ===")
    
    # Start the node
    node_process = start_lm_studio_node()
    
    try:
        # Run client tests (these will connect to the running node)
        from test_lm_studio_client import run_all_tests
        
        print("\n" + "="*50)
        print("Running client tests against live node...")
        print("="*50)
        
        client_success = run_all_tests()
        
        if client_success:
            print("✓ Client tests passed against live node")
        else:
            print("✗ Client tests failed against live node")
        
        return client_success
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False
    finally:
        stop_node(node_process)

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
