#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
import threading
import time
import json
from typing import Dict, Any, List

from std_msgs.msg import String
from sensor_msgs.msg import Image
from lm_studio_interfaces.action import ChatCompletion, TextCompletion
from lm_studio_interfaces.srv import ListModels, GetEmbeddings, SetModel, ResetChat

class LMStudioTester(Node):
    def __init__(self):
        super().__init__('lm_studio_tester')
        
        # Action clients
        self.chat_action_client = ActionClient(self, ChatCompletion, 'chat_completion')
        self.completion_action_client = ActionClient(self, TextCompletion, 'text_completion')
        
        # Service clients
        self.list_models_client = self.create_client(ListModels, 'list_models')
        self.embeddings_client = self.create_client(GetEmbeddings, 'get_embeddings')
        self.set_model_client = self.create_client(SetModel, 'set_model')
        self.reset_chat_client = self.create_client(ResetChat, 'reset_chat')
        
        # Subscribers for monitoring
        self.response_subscriber = self.create_subscription(
            String, 'lm_text_response', self.response_callback, 10
        )
        
        self.status_subscriber = self.create_subscription(
            String, 'lm_studio_status', self.status_callback, 10
        )
        
        self.models_subscriber = self.create_subscription(
            String, 'available_models', self.models_callback, 10
        )
        
        self.latest_response = None
        self.latest_status = None
        self.latest_models = None
        
        self.get_logger().info("LM Studio Tester initialized")

    def response_callback(self, msg):
        """Callback for text responses"""
        self.latest_response = json.loads(msg.data)
        self.get_logger().info(f"Received response: {self.latest_response['text'][:100]}...")

    def status_callback(self, msg):
        """Callback for status updates"""
        self.latest_status = json.loads(msg.data)
        self.get_logger().info(f"Status update: {self.latest_status}")

    def models_callback(self, msg):
        """Callback for models list updates"""
        self.latest_models = json.loads(msg.data)
        self.get_logger().info(f"Models update: {self.latest_models['count']} models available")

    def wait_for_service(self, client, service_name, timeout=10):
        """Wait for a service to become available"""
        start_time = time.time()
        while not client.wait_for_service(timeout_sec=1):
            if time.time() - start_time > timeout:
                self.get_logger().error(f"Service {service_name} not available")
                return False
            self.get_logger().info(f"Waiting for {service_name} service...")
        return True

    def test_list_models_service(self):
        """Test the list models service"""
        print("\n=== Testing List Models Service ===")
        
        if not self.wait_for_service(self.list_models_client, 'list_models'):
            return False
        
        request = ListModels.Request()
        future = self.list_models_client.call_async(request)
        
        # Wait for response
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            print(f"Service returned {response.count} models")
            for i, model in enumerate(response.models[:3]):  # Show first 3
                print(f"  {i+1}. {model}")
            return response.success
        else:
            print("Service call failed")
            return False

    def test_embeddings_service(self):
        """Test the embeddings service"""
        print("\n=== Testing Embeddings Service ===")
        
        if not self.wait_for_service(self.embeddings_client, 'get_embeddings'):
            return False
        
        request = GetEmbeddings.Request()
        request.text = "This is a test sentence for embeddings."
        request.timeout = 30
        
        future = self.embeddings_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            print(f"Generated embeddings with {response.dimensions} dimensions")
            return response.success
        else:
            print("Service call failed")
            return False

    def test_set_model_service(self):
        """Test the set model service"""
        print("\n=== Testing Set Model Service ===")
        
        if not self.wait_for_service(self.set_model_client, 'set_model'):
            return False
        
        # First get current models to have a valid model name
        list_request = ListModels.Request()
        list_future = self.list_models_client.call_async(list_request)
        rclpy.spin_until_future_complete(self, list_future)
        
        if list_future.result() is None or not list_future.result().success:
            print("Cannot test set model without available models")
            return False
        
        available_models = list_future.result().models
        if not available_models:
            print("No models available to test set model service")
            return False
        
        # Try to set to the first available model
        request = SetModel.Request()
        request.model_name = available_models[0]
        
        future = self.set_model_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            print(f"Set model result: {response.message}")
            return response.success
        else:
            print("Service call failed")
            return False

    def test_reset_chat_service(self):
        """Test the reset chat service"""
        print("\n=== Testing Reset Chat Service ===")
        
        if not self.wait_for_service(self.reset_chat_client, 'reset_chat'):
            return False
        
        request = ResetChat.Request()
        future = self.reset_chat_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            print(f"Reset chat result: {response.message}")
            return response.success
        else:
            print("Service call failed")
            return False

    def test_chat_action_standard(self):
        """Test standard chat completion action"""
        print("\n=== Testing Standard Chat Action ===")
        
        if not self.chat_action_client.wait_for_server(timeout_sec=10):
            print("Chat action server not available")
            return False
        
        goal_msg = ChatCompletion.Goal()
        goal_msg.prompt = "Hello! What can you tell me about robotics?"
        goal_msg.model = ""  # Use default
        goal_msg.max_tokens = 100
        goal_msg.temperature = 0.7
        goal_msg.timeout = 30
        goal_msg.use_history = True
        goal_msg.stream = False
        goal_msg.progress_feedback = False
        
        future = self.chat_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.chat_goal_response_callback)
        
        # Wait for result
        start_time = time.time()
        while not hasattr(self, 'chat_result') and time.time() - start_time < 60:
            rclpy.spin_once(self, timeout_sec=1)
        
        if hasattr(self, 'chat_result'):
            result = self.chat_result
            delattr(self, 'chat_result')
            print(f"Chat action completed: {result.success}")
            if result.success:
                print(f"Response: {result.response}")
            return result.success
        else:
            print("Chat action timed out")
            return False

    def test_chat_action_streaming(self):
        """Test streaming chat completion action"""
        print("\n=== Testing Streaming Chat Action ===")
        
        if not self.chat_action_client.wait_for_server(timeout_sec=10):
            print("Chat action server not available")
            return False
        
        goal_msg = ChatCompletion.Goal()
        goal_msg.prompt = "Explain artificial intelligence in one sentence."
        goal_msg.model = ""
        goal_msg.max_tokens = 50
        goal_msg.temperature = 0.7
        goal_msg.timeout = 30
        goal_msg.use_history = False
        goal_msg.stream = True
        goal_msg.progress_feedback = False
        
        self.streaming_feedback = []
        
        future = self.chat_action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.chat_feedback_callback
        )
        future.add_done_callback(self.chat_goal_response_callback)
        
        # Wait for result
        start_time = time.time()
        while not hasattr(self, 'chat_result') and time.time() - start_time < 60:
            rclpy.spin_once(self, timeout_sec=1)
        
        if hasattr(self, 'chat_result'):
            result = self.chat_result
            delattr(self, 'chat_result')
            print(f"Streaming chat completed: {result.success}")
            print(f"Received {len(self.streaming_feedback)} feedback messages")
            print(f"Final response: {result.response}")
            return result.success
        else:
            print("Streaming chat action timed out")
            return False

    def chat_goal_response_callback(self, future):
        """Callback for chat goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Chat goal rejected')
            self.chat_result = ChatCompletion.Result()
            self.chat_result.success = False
            return
        
        print('Chat goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.chat_result_callback)

    def chat_result_callback(self, future):
        """Callback for chat result"""
        result = future.result().result
        self.chat_result = result

    def chat_feedback_callback(self, feedback_msg):
        """Callback for chat feedback during streaming"""
        self.streaming_feedback.append(feedback_msg)
        if feedback_msg.partial_response:
            print(feedback_msg.partial_response, end='', flush=True)

    def test_text_completion_action(self):
        """Test text completion action"""
        print("\n=== Testing Text Completion Action ===")
        
        if not self.completion_action_client.wait_for_server(timeout_sec=10):
            print("Completion action server not available")
            return False
        
        goal_msg = TextCompletion.Goal()
        goal_msg.prompt = "The future of autonomous vehicles"
        goal_msg.model = ""
        goal_msg.max_tokens = 50
        goal_msg.temperature = 0.7
        goal_msg.timeout = 30
        goal_msg.stream = False
        goal_msg.progress_feedback = False
        
        future = self.completion_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.completion_goal_response_callback)
        
        # Wait for result
        start_time = time.time()
        while not hasattr(self, 'completion_result') and time.time() - start_time < 60:
            rclpy.spin_once(self, timeout_sec=1)
        
        if hasattr(self, 'completion_result'):
            result = self.completion_result
            delattr(self, 'completion_result')
            print(f"Text completion completed: {result.success}")
            if result.success:
                print(f"Response: {result.response}")
            return result.success
        else:
            print("Text completion action timed out")
            return False

    def completion_goal_response_callback(self, future):
        """Callback for completion goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Completion goal rejected')
            self.completion_result = TextCompletion.Result()
            self.completion_result.success = False
            return
        
        print('Completion goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.completion_result_callback)

    def completion_result_callback(self, future):
        """Callback for completion result"""
        result = future.result().result
        self.completion_result = result

    def run_all_tests(self):
        """Run all ROS node tests"""
        print("Starting LM Studio ROS Node Tests...")
        
        # Wait a bit for node to initialize
        time.sleep(2.0)
        
        tests = [
            ("List Models Service", self.test_list_models_service),
            ("Embeddings Service", self.test_embeddings_service),
            ("Set Model Service", self.test_set_model_service),
            ("Reset Chat Service", self.test_reset_chat_service),
            ("Standard Chat Action", self.test_chat_action_standard),
            ("Streaming Chat Action", self.test_chat_action_streaming),
            ("Text Completion Action", self.test_text_completion_action),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"{status}: {test_name}\n")
            except Exception as e:
                print(f"✗ ERROR: {test_name} - {e}\n")
                results.append((test_name, False))
            
            time.sleep(1.0)  # Brief pause between tests
        
        # Summary
        print("=== TEST SUMMARY ===")
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        return passed == total

def main():
    rclpy.init()
    
    tester = LMStudioTester()
    
    try:
        # Run tests in a separate thread to allow spinning
        test_thread = threading.Thread(target=tester.run_all_tests)
        test_thread.start()
        
        # Spin to process callbacks
        rclpy.spin(tester)
        
        test_thread.join()
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
