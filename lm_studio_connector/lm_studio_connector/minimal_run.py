#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import threading
import time

from lm_studio_interfaces.action import ChatCompletion

class LMStudioChatClient(Node):
    def __init__(self):
        super().__init__('lm_studio_chat_client')
        
        # Create action client
        self.action_client = ActionClient(self, ChatCompletion, 'chat_completion')
        
        self.get_logger().info("LM Studio Chat Client initialized")

    def send_chat_request(self, prompt, use_history=True, stream=False, progress_feedback=False):
        """Send a chat request and handle feedback/results"""
        
        # Wait for action server to be available
        self.get_logger().info("Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available")
            return False

        # Create goal message
        goal_msg = ChatCompletion.Goal()
        goal_msg.prompt = prompt
        goal_msg.use_history = use_history
        goal_msg.stream = stream
        goal_msg.progress_feedback = progress_feedback
        goal_msg.image_data = ""  # Empty for text-only chat
        goal_msg.model = ""  # Use default model
        goal_msg.max_tokens = 500
        goal_msg.temperature = 0.7
        goal_msg.timeout = 30

        self.get_logger().info(f"Sending chat request: '{prompt}'")
        
        # Send goal and get future
        goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        # Add callback for when goal is accepted
        goal_future.add_done_callback(self.goal_response_callback)
        
        return True

    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected by server"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        
        # Get result future
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Callback for receiving feedback during execution"""
        feedback = feedback_msg.feedback
        
#        if feedback.partial_response:
#            # Streaming mode: print each token as it arrives
#            print(feedback.partial_response, end='', flush=True)
#        
        self.get_logger().info(
            f'Feedback: Status="{feedback.status}", Progress={feedback.progress:.2f}'
        )

    def result_callback(self, future):
        """Callback when action is completed"""
        result = future.result().result
        
        if result.success:
            self.get_logger().info('Action completed successfully!')
            self.get_logger().info(f'Response: {result.response}')
            
            # Print the full response (useful for non-streaming mode)
            #if result.response:
            #    print(f"\n\nFull response: {result.response}")
        else:
            self.get_logger().error(f'Action failed: {result.response}')


def main():
    rclpy.init()
    
    # Create client node
    client = LMStudioChatClient()
    
    # Start ROS spinning in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(client,), daemon=True)
    spin_thread.start()
    
    try:
        # Example 1: Standard chat (non-streaming)
        print("\n=== Example 1: Standard Chat ===")
        client.send_chat_request(
            prompt="Hello! Tell me a short joke about robots.",
            use_history=True,
            stream=False,
            progress_feedback=False
        )
        
        # Wait for completion
        time.sleep(10)
        
        # Example 2: Streaming chat
        print("\n\n=== Example 2: Streaming Chat ===")
        print("Streaming response: ", end='')
        client.send_chat_request(
            prompt="Explain the concept of artificial intelligence in one sentence.",
            use_history=True,
            stream=True,  # Enable streaming
            progress_feedback=False
        )
        
        # Wait for completion
        time.sleep(10)
        
        # Example 3: Chat with progress feedback
        print("\n\n=== Example 3: Chat with Progress Feedback ===")
        client.send_chat_request(
            prompt="Write a short story about a robot exploring Mars.",
            use_history=True,
            stream=False,
            progress_feedback=True  # Enable progress updates
        )
        
        # Keep running to receive all callbacks
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
