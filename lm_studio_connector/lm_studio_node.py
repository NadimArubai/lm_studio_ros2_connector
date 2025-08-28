#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any, List

from .lm_studio_client import LMStudioClient

class LMStudioNode(Node):
    """
    ROS2 Node optimized for LM Studio API with image support
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
        self.declare_parameter('stream', False)
        self.declare_parameter('max_history_length', 10)
        
        # Get parameters
        lm_studio_url = self.get_parameter('lm_studio_url').value
        api_key = self.get_parameter('api_key').value
        self.model_name = self.get_parameter('model_name').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.temperature = self.get_parameter('temperature').value
        self.timeout = self.get_parameter('timeout').value
        self.stream = self.get_parameter('stream').value
        self.max_history_length = self.get_parameter('max_history_length').value
        
        # Initialize LM Studio client
        self.lm_client = LMStudioClient(lm_studio_url, api_key)
        
        # Conversation history for chat mode
        self.conversation_history = []
        
        # Available models
        self.available_models = []
        
        # Latest image handle
        self.latest_image_handle = None
        
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
        
        # Subscribers
        self.create_subscription(
            String,
            'chat_input',
            self.chat_input_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.create_subscription(
            String,
            'completion_input',
            self.completion_input_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.create_subscription(
            Image,
            'image_input',
            self.image_input_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.create_subscription(
            String,
            'chat_with_image',
            self.chat_with_image_callback,
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
        
        self.create_subscription(
            String,
            'get_embeddings',
            self.embeddings_callback,
            10
        )
        
        self.create_subscription(
            String,
            'set_model',
            self.set_model_callback,
            10
        )
        
        self.create_subscription(
            String,
            'reset_chat',
            self.reset_chat_callback,
            10
        )
        
        # Check connection and get available models
        self.initialize_connection()
        
        # Timers
        self.create_timer(60000.0, self.models_update_timer)  # Update models periodically
        self.create_timer(10.0, self.status_timer_callback)
        
        self.get_logger().info("LM Studio Node initialized with image support")
        

    def initialize_connection(self):
        """Initialize connection and get available models"""
        try:
            # Test connection
            connection_status = self.lm_client.test_connection()
            self.get_logger().info(f"Connection test: {json.dumps(connection_status, indent=2)}")
            
            # Get available models
            self.available_models = self.lm_client.list_models()
            self.publish_models_list()
            
            if connection_status["health_status"]:
                self.get_logger().info("Connected to LM Studio successfully")
            else:
                self.get_logger().warn("LM Studio connection issues, but node will continue")
                
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
    
    
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
            
    def image_input_callback(self, msg: Image):
        """Handle raw image input and convert to base64 for LM Studio"""
        try:
            self.get_logger().info("Received image input")
            
            # Convert ROS Image message to OpenCV format
            cv_image = self.image_msg_to_cv2(msg)
            
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
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def chat_with_image_callback(self, msg: String):
        """Handle chat input with the latest available image"""
        try:
            self.get_logger().info(f"Received chat with image input: {msg.data}")
            
            if not self.latest_image_handle:
                self.get_logger().warn("No image available for chat")
                self.publish_text_response("No image available. Please send an image first.", "error")
                return
            
            # Create multimodal message
            message = {
                "role": "user",
                "content": msg.data,
                "images": [self.latest_image_handle]
            }
            
#            self.get_logger().info(f"Sending message with image: {json.dumps(message, indent=2)}")
            self.get_logger().info(f"Sending message with image")#: {json.dumps(message, indent=2)}")
            
            # Generate response using chat completion
            response = self.lm_client.chat_completion(
                messages=[message],  # Send only this message for now
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=self.stream,
                timeout=self.timeout
            )
            
            # Extract response text
            if 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['message']['content']
                self.get_logger().info(f"Model response: {response_text}")
            else:
                response_text = "No response generated"
                self.get_logger().warn("No choices in response")
            
            # Publish response
            self.publish_text_response(response_text, "chat_with_image")
            
        except Exception as e:
            self.get_logger().error(f"Error processing chat with image: {e}")
            self.publish_text_response(f"Error: {str(e)}", "error")
    
    
    def chat_input_callback(self, msg: String):
        """Handle chat input using /v1/chat/completions"""
        try:
            self.get_logger().info(f"Received chat input: {msg.data}")
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": msg.data})
            
            # Keep history within limits
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Generate response using chat completion
            response = self.lm_client.chat_completion(
                messages=self.conversation_history,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=self.stream,
                timeout=self.timeout
            )
            
            # Extract response text
            if 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['message']['content']
                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
            else:
                response_text = "No response generated"
            
            # Publish response
            self.publish_text_response(response_text, "chat")
            
        except Exception as e:
            self.get_logger().error(f"Error processing chat input: {e}")
            self.publish_text_response(f"Error: {str(e)}", "error")
    
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



    def completion_input_callback(self, msg: String):
        """Handle completion input using /v1/completions"""
        try:
            self.get_logger().info(f"Received completion input: {msg.data}")
            
            # Generate response using text completion
            response = self.lm_client.generate_text(
                prompt=msg.data,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=self.stream,
                timeout=self.timeout
            )
            
            # Extract response text
            if 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['text']
            else:
                response_text = "No response generated"
            
            # Publish response
            self.publish_text_response(response_text, "completion")
            
        except Exception as e:
            self.get_logger().error(f"Error processing completion input: {e}")
            self.publish_text_response(f"Error: {str(e)}", "error")
    
    def embeddings_callback(self, msg: String):
        """Handle embeddings requests"""
        try:
            self.get_logger().info(f"Getting embeddings for: {msg.data}")
            
            embeddings = self.lm_client.get_embeddings(
                input_text=msg.data,
                model=self.model_name,
                timeout=self.timeout
            )
            
            # Publish embeddings (truncated for logging)
            self.get_logger().info(f"Got embeddings of length {len(embeddings)}")
            
            # You could publish embeddings to a topic if needed
            # self.publish_embeddings(embeddings)
            
        except Exception as e:
            self.get_logger().error(f"Error getting embeddings: {e}")
    
    def set_model_callback(self, msg: String):
        """Change the active model"""
        new_model = msg.data
        if new_model:
            self.model_name = new_model
            self.get_logger().info(f"Changed model to: {self.model_name}")
            self.publish_text_response(f"Model changed to {self.model_name}", "system")

    def reset_chat_callback(self, msg: String):
        """Reset conversation history"""
        self.conversation_history = []
        self.get_logger().info("Chat history reset")
        self.publish_text_response("Chat history reset", "system")
    
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
            self.get_logger().info(f"Published {response_type} response: {response_text[:100]}...")
            
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
