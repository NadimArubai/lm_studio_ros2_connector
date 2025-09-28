#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from lm_studio_interfaces.action import ChatCompletion

class MinimalChatClient(Node):
    def __init__(self):
        super().__init__('minimal_chat_client')
        self.action_client = ActionClient(self, ChatCompletion, 'chat_completion')

    def chat(self, prompt):
        """Minimal chat example with callbacks"""
        
        # Wait for server
        self.action_client.wait_for_server()
        
        # Create goal
        goal_msg = ChatCompletion.Goal()
        goal_msg.prompt = prompt
        goal_msg.use_history = True
        goal_msg.stream = False # Try to toggle this
        
        # Send goal with callbacks
        self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb
        ).add_done_callback(self._goal_response_cb)
        
        self.get_logger().info(f"Sent: {prompt}")

    def _goal_response_cb(self, future):
        """Handle goal acceptance"""
        goal_handle = future.result()
        if goal_handle.accepted:
            goal_handle.get_result_async().add_done_callback(self._result_cb)
            self.get_logger().info("Goal accepted")

    def _feedback_cb(self, feedback_msg):
        """Handle feedback"""
        feedback = feedback_msg.feedback
        if feedback.partial_response:
            print(feedback.partial_response, end='', flush=True)

    def _result_cb(self, future):
        """Handle final result"""
        result = future.result().result
        if result.success:
            self.get_logger().info(f"Success: {result.response}")
        else:
            self.get_logger().error(f"Failed: {result.response}")
        
        # Shutdown after receiving result
        rclpy.shutdown()


def main():
    rclpy.init()
    client = MinimalChatClient()
    
    # Send a chat message
    client.chat("Hello, how are you?")
    
    # Keep spinning to receive callbacks
    rclpy.spin(client)

if __name__ == '__main__':
    main()
