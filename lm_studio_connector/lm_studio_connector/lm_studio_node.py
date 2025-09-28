#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from lm_studio_interfaces.action import ChatCompletion, TextCompletion
from lm_studio_interfaces.srv import ListModels, GetEmbeddings, SetModel, ResetChat
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any, List
import os
import time

from lm_studio_connector.lm_studio_client import LMStudioClient

class LMStudioNode(Node):
    """
    ROS2 Node optimized for LM Studio API with image support and streaming
    """
    
    def __init__(self):
        super().__init__('lm_studio_node')
        
        # Parameters for LM Studio
        self.declare_parameter('lm_studio_url', 'http://localhost:1234')
        self.declare_parameter('api_key', '')
        self.declare_parameter('model_name', 'local-model')
        self.declare_parameter('max_tokens', 500)
        self.declare_parameter('temperature', 0.7)
        self.declare_parameter('timeout', 30)
        self.declare_parameter('max_history_length', 10)
        
        # Get parameters
        lm_studio_url = self.get_parameter('lm_studio_url').value
        api_key = self.get_parameter('api_key').value
        self.model_name = self.get_parameter('model_name').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.temperature = self.get_parameter('temperature').value
        self.timeout = self.get_parameter('timeout').value
        self.max_history_length = self.get_parameter('max_history_length').value
        
        # Initialize LM Studio client
        self.lm_client = LMStudioClient(lm_studio_url, api_key)
        
        # Conversation history for chat mode
        self.conversation_history = []
        
        # Available models
        self.available_models = []
        
        # Latest image handle
        self.latest_image_handle = None
        
        # Bridge for compressed images
        self.bridge = CvBridge()
        
        # Publishers
        self.text_response_pub = self.create_publisher(
            String,
            'lm_text_response',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            'lm_studio_status',
            10
        )
        
        self.models_pub = self.create_publisher(
            String,
            'available_models',
            10
        )
        
        # Action Servers (for long-running tasks)
        self.chat_action_server = ActionServer(
            self,
            ChatCompletion,
            'chat_completion',
            self.execute_chat_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.completion_action_server = ActionServer(
            self,
            TextCompletion,
            'text_completion',
            self.execute_completion_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Service Servers (for quick requests) - unchanged
        self.models_service = self.create_service(
            ListModels,
            'list_models',
            self.list_models_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.embeddings_service = self.create_service(
            GetEmbeddings,
            'get_embeddings',
            self.get_embeddings_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.set_model_service = self.create_service(
            SetModel,
            'set_model',
            self.set_model_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.reset_chat_service = self.create_service(
            ResetChat,
            'reset_chat',
            self.reset_chat_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Subscribers (for image inputs) - unchanged
        self.create_subscription(
            Image,
            'image_input',
            self.image_input_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.create_subscription(
            CompressedImage,
            'image_input/compressed',
            self.compressed_image_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.create_subscription(
            String,
            'image_file_input',
            self.image_file_input_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Check connection and get available models
        self.initialize_connection()
        
        # Timers
        self.create_timer(10.0, self.models_update_timer)
        self.create_timer(10.0, self.status_timer_callback)
        
        self.get_logger().info("LM Studio Node initialized with streaming support")

    def initialize_connection(self):
        """Initialize connection and get available models"""
        try:
            connection_status = self.lm_client.test_connection()
            self.get_logger().info(f"Connection test: {json.dumps(connection_status, indent=2)}")
            
            self.available_models = self.lm_client.list_models()
            self.publish_models_list()
            
            if connection_status["health_status"]:
                self.get_logger().info("Connected to LM Studio successfully")
            else:
                self.get_logger().warn("LM Studio connection issues, but node will continue")
                
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")

    def parse_image_data(self, image_data):
        """Parse image data and return image handle"""
        image_handle = None
        
        if image_data == "latest":
            image_handle = self.latest_image_handle
            
        elif image_data.startswith("data:image/"):
            base64_data = image_data.split("base64,")[1]
            image_handle = self.lm_client.prepare_image_base64(base64_data)
            
        elif image_data and os.path.exists(image_data):
            image_handle = self.lm_client.prepare_image(image_data)
            
        return image_handle

    def execute_chat_callback(self, goal_handle):
        """Action server callback for chat completion with streaming support"""
        try:
            request = goal_handle.request
            self.get_logger().info(f"Received chat action request: stream={request.stream}, progress_feedback={request.progress_feedback}")
            
            # Handle different modes based on request flags
            if request.stream:
                return self._handle_streaming_chat(goal_handle)
            elif request.progress_feedback:
                return self._handle_progress_chat(goal_handle)
            else:
                return self._handle_standard_chat(goal_handle)
                
        except Exception as e:
            self.get_logger().error(f"Chat action failed: {e}")
            result = ChatCompletion.Result()
            result.response = f"Error: {str(e)}"
            result.success = False
            goal_handle.abort()
            return result

    def _handle_standard_chat(self, goal_handle):
        """Handle standard non-streaming chat completion"""
        request = goal_handle.request
        
        # Prepare messages
        messages = []
        if request.use_history:
            messages.extend(self.conversation_history)
        
        user_message = {"role": "user", "content": request.prompt}
        image_handle = self.parse_image_data(request.image_data)
        if image_handle:
            user_message["images"] = [image_handle]
        
        messages.append(user_message)
        
        # Generate response
        response = self.lm_client.chat_completion(
            messages=messages,
            model=request.model if request.model else self.model_name,
            max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
            temperature=request.temperature if request.temperature >= 0 else self.temperature,
            stream=False,
            timeout=request.timeout if request.timeout > 0 else self.timeout
        )
        
        # Extract response
        if 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['message']['content']
            
            if request.use_history:
                self.conversation_history.append(user_message)
                self.conversation_history.append({"role": "assistant", "content": response_text})
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
        else:
            response_text = "No response generated"
        
        # Publish response
        self.publish_text_response(response_text, "chat_action")
        
        # Return result
        result = ChatCompletion.Result()
        result.response = response_text
        result.success = True
        
        goal_handle.succeed()
        return result

    def _handle_streaming_chat(self, goal_handle):
        """Handle streaming chat completion with real-time token feedback"""
        request = goal_handle.request
        
        # Prepare messages
        messages = []
        if request.use_history:
            messages.extend(self.conversation_history)
        
        user_message = {"role": "user", "content": request.prompt}
        image_handle = self.parse_image_data(request.image_data)
        if image_handle:
            user_message["images"] = [image_handle]
        
        messages.append(user_message)
        
        # Get streaming response
        stream = self.lm_client.chat_completion(
            messages=messages,
            model=request.model if request.model else self.model_name,
            max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
            temperature=request.temperature if request.temperature >= 0 else self.temperature,
            stream=True,
            timeout=request.timeout if request.timeout > 0 else self.timeout
        )
        
        full_response = ""
        try:
            for chunk in stream:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return ChatCompletion.Result()
                
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        token = delta['content']
                        full_response += token
                        
                        # Publish feedback for each token
                        feedback_msg = ChatCompletion.Feedback()
                        feedback_msg.partial_response = token
                        feedback_msg.status = "streaming"
                        feedback_msg.progress = len(full_response) / request.max_tokens if request.max_tokens > 0 else 0.0
                        goal_handle.publish_feedback(feedback_msg)
            
            # Final response
            if request.use_history:
                self.conversation_history.append(user_message)
                self.conversation_history.append({"role": "assistant", "content": full_response})
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            self.publish_text_response(full_response, "chat_action_stream")
            
            result = ChatCompletion.Result()
            result.response = full_response
            result.success = True
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.get_logger().error(f"Streaming chat failed: {e}")
            result = ChatCompletion.Result()
            result.response = f"Error during streaming: {str(e)}"
            result.success = False
            goal_handle.abort()
            return result

    def _handle_progress_chat(self, goal_handle):
        """Handle chat completion with periodic progress feedback"""
        request = goal_handle.request
        
        # Prepare messages
        messages = []
        if request.use_history:
            messages.extend(self.conversation_history)
        
        user_message = {"role": "user", "content": request.prompt}
        image_handle = self.parse_image_data(request.image_data)
        if image_handle:
            user_message["images"] = [image_handle]
        
        messages.append(user_message)
        
        # Start the request in a separate thread (simplified)
        # In practice, you might need more sophisticated async handling
        import threading
        
        response_result = {"text": "", "error": None}
        
        def make_request():
            try:
                response = self.lm_client.chat_completion(
                    messages=messages,
                    model=request.model if request.model else self.model_name,
                    max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
                    temperature=request.temperature if request.temperature >= 0 else self.temperature,
                    stream=False,
                    timeout=request.timeout if request.timeout > 0 else self.timeout
                )
                
                if 'choices' in response and len(response['choices']) > 0:
                    response_result["text"] = response['choices'][0]['message']['content']
                else:
                    response_result["text"] = "No response generated"
                    
            except Exception as e:
                response_result["error"] = str(e)
        
        # Start the request thread
        request_thread = threading.Thread(target=make_request)
        request_thread.start()
        
        # Send periodic progress updates
        start_time = time.time()
        while request_thread.is_alive():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return ChatCompletion.Result()
            
            elapsed = time.time() - start_time
            progress = min(elapsed / request.timeout, 0.99) if request.timeout > 0 else 0.5
            
            feedback_msg = ChatCompletion.Feedback()
            feedback_msg.partial_response = ""
            feedback_msg.status = f"Processing... ({elapsed:.1f}s elapsed)"
            feedback_msg.progress = progress
            goal_handle.publish_feedback(feedback_msg)
            
            time.sleep(1.0)  # Update every second
        
        # Wait for thread to complete
        request_thread.join()
        
        # Handle result
        if response_result["error"]:
            result = ChatCompletion.Result()
            result.response = f"Error: {response_result['error']}"
            result.success = False
            goal_handle.abort()
            return result
        
        response_text = response_result["text"]
        
        if request.use_history:
            self.conversation_history.append(user_message)
            self.conversation_history.append({"role": "assistant", "content": response_text})
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        self.publish_text_response(response_text, "chat_action_progress")
        
        result = ChatCompletion.Result()
        result.response = response_text
        result.success = True
        goal_handle.succeed()
        return result

    def execute_completion_callback(self, goal_handle):
        """Action server callback for text completion with streaming support"""
        try:
            request = goal_handle.request
            self.get_logger().info(f"Received completion action request: stream={request.stream}, progress_feedback={request.progress_feedback}")
            
            # Handle different modes based on request flags
            if request.stream:
                return self._handle_streaming_completion(goal_handle)
            elif request.progress_feedback:
                return self._handle_progress_completion(goal_handle)
            else:
                return self._handle_standard_completion(goal_handle)
                
        except Exception as e:
            self.get_logger().error(f"Completion action failed: {e}")
            result = TextCompletion.Result()
            result.response = f"Error: {str(e)}"
            result.success = False
            goal_handle.abort()
            return result

    def _handle_standard_completion(self, goal_handle):
        """Handle standard non-streaming text completion"""
        request = goal_handle.request
        
        response = self.lm_client.generate_text(
            prompt=request.prompt,
            model=request.model if request.model else self.model_name,
            max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
            temperature=request.temperature if request.temperature >= 0 else self.temperature,
            stream=False,
            timeout=request.timeout if request.timeout > 0 else self.timeout
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text']
        else:
            response_text = "No response generated"
        
        self.publish_text_response(response_text, "completion_action")
        
        result = TextCompletion.Result()
        result.response = response_text
        result.success = True
        
        goal_handle.succeed()
        return result

    def _handle_streaming_completion(self, goal_handle):
        """Handle streaming text completion with real-time token feedback"""
        request = goal_handle.request
        
        stream = self.lm_client.generate_text(
            prompt=request.prompt,
            model=request.model if request.model else self.model_name,
            max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
            temperature=request.temperature if request.temperature >= 0 else self.temperature,
            stream=True,
            timeout=request.timeout if request.timeout > 0 else self.timeout
        )
        
        full_response = ""
        try:
            for chunk in stream:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return TextCompletion.Result()
                
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    if text:
                        full_response += text
                        
                        feedback_msg = TextCompletion.Feedback()
                        feedback_msg.partial_response = text
                        feedback_msg.status = "streaming"
                        feedback_msg.progress = len(full_response) / request.max_tokens if request.max_tokens > 0 else 0.0
                        goal_handle.publish_feedback(feedback_msg)
            
            self.publish_text_response(full_response, "completion_action_stream")
            
            result = TextCompletion.Result()
            result.response = full_response
            result.success = True
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.get_logger().error(f"Streaming completion failed: {e}")
            result = TextCompletion.Result()
            result.response = f"Error during streaming: {str(e)}"
            result.success = False
            goal_handle.abort()
            return result

    def _handle_progress_completion(self, goal_handle):
        """Handle text completion with periodic progress feedback"""
        request = goal_handle.request
        
        import threading
        
        response_result = {"text": "", "error": None}
        
        def make_request():
            try:
                response = self.lm_client.generate_text(
                    prompt=request.prompt,
                    model=request.model if request.model else self.model_name,
                    max_tokens=request.max_tokens if request.max_tokens > 0 else self.max_tokens,
                    temperature=request.temperature if request.temperature >= 0 else self.temperature,
                    stream=False,
                    timeout=request.timeout if request.timeout > 0 else self.timeout
                )
                
                if 'choices' in response and len(response['choices']) > 0:
                    response_result["text"] = response['choices'][0]['text']
                else:
                    response_result["text"] = "No response generated"
                    
            except Exception as e:
                response_result["error"] = str(e)
        
        request_thread = threading.Thread(target=make_request)
        request_thread.start()
        
        start_time = time.time()
        while request_thread.is_alive():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return TextCompletion.Result()
            
            elapsed = time.time() - start_time
            progress = min(elapsed / request.timeout, 0.99) if request.timeout > 0 else 0.5
            
            feedback_msg = TextCompletion.Feedback()
            feedback_msg.partial_response = ""
            feedback_msg.status = f"Processing... ({elapsed:.1f}s elapsed)"
            feedback_msg.progress = progress
            goal_handle.publish_feedback(feedback_msg)
            
            time.sleep(1.0)
        
        request_thread.join()
        
        if response_result["error"]:
            result = TextCompletion.Result()
            result.response = f"Error: {response_result['error']}"
            result.success = False
            goal_handle.abort()
            return result
        
        response_text = response_result["text"]
        self.publish_text_response(response_text, "completion_action_progress")
        
        result = TextCompletion.Result()
        result.response = response_text
        result.success = True
        goal_handle.succeed()
        return result


    
    def list_models_callback(self, request, response):
        """Service callback for listing models"""
        try:
            self.get_logger().info("Received list models service request")
            
            models = self.lm_client.list_models(timeout=self.timeout)
            response.models = [model.get('id', 'unknown') for model in models]
            response.count = len(models)
            response.success = True
            
            self.get_logger().info(f"Returning {response.count} models")
            return response
            
        except Exception as e:
            self.get_logger().error(f"List models service failed: {e}")
            response.success = False
            response.error_message = str(e)
            return response

    def get_embeddings_callback(self, request, response):
        """Service callback for getting embeddings"""
        try:
            self.get_logger().info(f"Received embeddings service request for: {request.text}")
            
            embeddings = self.lm_client.get_embeddings(
                input_text=request.text,
                model=request.model if request.model else self.model_name,
                timeout=request.timeout if request.timeout > 0 else self.timeout
            )
            
            response.embeddings = embeddings
            response.success = True
            response.dimensions = len(embeddings)
            
            self.get_logger().info(f"Returning embeddings with {response.dimensions} dimensions")
            return response
            
        except Exception as e:
            self.get_logger().error(f"Embeddings service failed: {e}")
            response.success = False
            response.error_message = str(e)
            return response

    def set_model_callback(self, request, response):
        """Service callback for setting model"""
        try:
            self.get_logger().info(f"Received set model service request: {request.model_name}")
            
            if request.model_name:
                self.model_name = request.model_name
                response.success = True
                response.message = f"Model changed to {self.model_name}"
                self.get_logger().info(response.message)
            else:
                response.success = False
                response.message = "Empty model name provided"
                
            return response
            
        except Exception as e:
            self.get_logger().error(f"Set model service failed: {e}")
            response.success = False
            response.message = str(e)
            return response

    def reset_chat_callback(self, request, response):
        """Service callback for resetting chat"""
        try:
            self.get_logger().info("Received reset chat service request")
            
            self.conversation_history = []
            response.success = True
            response.message = "Chat history reset"
            
            self.get_logger().info(response.message)
            return response
            
        except Exception as e:
            self.get_logger().error(f"Reset chat service failed: {e}")
            response.success = False
            response.message = str(e)
            return response

    # Image handling methods
    def image_file_input_callback(self, msg: String):
        """Handle image file path input"""
        try:
            image_path = msg.data.strip()
            self.get_logger().info(f"Received image file path: {image_path}")
            
            # Prepare image using file path
            image_handle = self.lm_client.prepare_image(image_path)
            
            if image_handle:
                self.latest_image_handle = image_handle
                self.get_logger().info(f"Image prepared from file: {image_path}")
                self.publish_text_response(f"Image loaded from {image_path}", "system")
            else:
                self.get_logger().error(f"Failed to prepare image from: {image_path}")
                self.publish_text_response(f"Failed to load image from {image_path}", "error")
                
        except Exception as e:
            self.get_logger().error(f"Error processing image file: {e}")
            self.publish_text_response(f"Error: {str(e)}", "error")
    
    def convert_cv_image_to_base64_and_store_it(self, cv_image):
        # Convert to JPEG and then to base64
            success, encoded_image = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                
                # Prepare image handle using data URI format
                image_handle = self.lm_client.prepare_image_base64(base64_image)
                
                if image_handle:
                    self.latest_image_handle = image_handle
                    self.get_logger().info("Image prepared from ROS message")
                else:
                    self.get_logger().error("Failed to prepare image handle")
                    
            else:
                self.get_logger().error("Failed to encode image to JPEG")   
    
    def image_input_callback(self, msg: Image):
        """Handle raw image input and convert to base64 for LM Studio"""
        try:
            self.get_logger().info("Received image input")
            
            # Convert ROS Image message to OpenCV format
            cv_image = self.image_msg_to_cv2(msg)
            
            # Convert to JPEG and then to base64
            self.convert_cv_image_to_base64_and_store_it(cv_image)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed image input and convert to base64 for LM Studio"""
        try:
            self.get_logger().info("Received compressed image input")
            
            # Convert ROS Compressed Image message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            
            # Convert to JPEG and then to base64
            self.convert_cv_image_to_base64_and_store_it(cv_image)
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            
        
    def image_msg_to_cv2(self, msg: Image) -> np.ndarray:
        """Convert ROS Image message to OpenCV format"""
        try:
            if msg.encoding == 'rgb8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
            elif msg.encoding == 'mono8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                # Try to convert anyway
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                
            return cv_image
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            raise

    def publish_models_list(self):
        """Publish list of available models"""
        try:
            models_msg = String()
            models_data = {
                "models": [model.get('id', 'unknown') for model in self.available_models],
                "count": len(self.available_models),
                "current_model": self.model_name
            }
            models_msg.data = json.dumps(models_data)
            self.models_pub.publish(models_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing models list: {e}")
    
    def publish_text_response(self, response_text: str, response_type: str = "response"):
        """Publish text response with metadata"""
        try:
            response_msg = String()
            response_data = {
                "text": response_text,
                "type": response_type,
                "model": self.model_name,
                "timestamp": self.get_clock().now().to_msg().sec,
                "history_length": len(self.conversation_history)
            }
            response_msg.data = json.dumps(response_data)
            
            self.text_response_pub.publish(response_msg)
            self.get_logger().info(f"Published {response_type} response")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing response: {e}")
    
    def status_timer_callback(self):
        """Periodic status update"""
        try:
            status_msg = String()
            status_data = {
                "model": self.model_name,
                "chat_history_length": len(self.conversation_history),
                "available_models": len(self.available_models),
                "health_status": self.lm_client.health_check()
            }
            status_msg.data = json.dumps(status_data)
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Status update failed: {e}")
    
    def models_update_timer(self):
        """Periodically update available models"""
        try:
            new_models = self.lm_client.list_models()
            if new_models != self.available_models:
                self.available_models = new_models
                self.publish_models_list()
                self.get_logger().info(f"Updated models list: {len(self.available_models)} models")
                
        except Exception as e:
            self.get_logger().error(f"Models update failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        executor = MultiThreadedExecutor()
        lm_studio_node = LMStudioNode()
        executor.add_node(lm_studio_node)
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node failed: {e}")
    finally:
        if 'lm_studio_node' in locals():
            lm_studio_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
