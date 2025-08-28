# LM Studio ROS2 Node

A ROS2 node for integrating LM Studio's local LLM API with robotics applications, supporting both text and image inputs for multimodal AI capabilities.

Features
🤖 **Chat Completion**: Text-based conversations with LLMs

🖼️ **Multimodal Support: Image input processing for Vision-Language Models (VLMs)

📝 **Text Completion**: Standard text generation

🔍 **Embeddings**: Text embedding generation

📊 **Model Management**: Dynamic model switching and discovery

⚡ **Real-time**: ROS2 topics for seamless integration

🏥 **Health Monitoring**: Connection status and health checks

# Prerequisites
ROS2 Humble or newer

Python 3.8+

LM Studio running locally (default: http://localhost:1234)

A Vision-Language Model (VLM) like qwen2-vl-2b-instruct for image support

# Installation
Install LM Studio: Download from lmstudio.ai

Download a VLM (for image support):

```bash
lms get qwen2-vl-2b-instruct
```

Add to your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone <this-repository>
cd ..
colcon build --packages-select lm_studio_node
source install/setup.bash
```

# Usage
Starting the Node

```bash
# Terminal 1: Start LM Studio and load your model
# (Open LM Studio GUI or use CLI)

# Terminal 2: Run the ROS2 node
ros2 run lm_studio_node lm_studio_nod
```

## Configuration Parameters
The node supports these parameters (set via launch file or command line):

|Parameter           |Default                 |Description                |
|--------------------|------------------------|---------------------------|
|lm_studio_url       |http://localhost:1234   |LM Studio API URL          |
|api_key             |""                      |API key (if required)      |
|model_name          |"local-model"           |Default model to use       |
|max_tokens          |500                     |Maximum response tokens    |
|temperature         |0.7                     |Creativity level (0.0-1.0) |
|timeout             |30                      |API timeout in seconds     |
|stream              |false                   |Stream responses           |
|max_history_length  |10                      |Chat history context length|

## Example Launch File
```xml
<launch>
    <node pkg="lm_studio_node" exec="lm_studio_node" name="lm_studio">
        <param name="lm_studio_url" value="http://localhost:1234"/>
        <param name="model_name" value="qwen2-vl-2b-instruct"/>
        <param name="max_tokens" value="1000"/>
        <param name="temperature" value="0.3"/>
    </node>
</launch>
```
## Topics
- Subscribers (Input)

|Topic              |Message Type       |Description                  |
|-------------------|-------------------|-----------------------------|
|/chat_input        |std_msgs/String    |Text chat messages           |
|/completion_input  |std_msgs/String    |Text completion prompts      |
|/image_input       |sensor_msgs/Image  |Raw image data               |
|/image_file_input  |std_msgs/String    |Path to image file           |
|/chat_with_image   |std_msgs/String    |Text prompt for current image|
|/get_embeddings    |std_msgs/String    |Text to generate embeddings  |
|/set_model         |std_msgs/String    |Change active model          |
|/reset_chat        |std_msgs/String    |Clear conversation history   |

- Publishers (Output)

|Topic              |Message Type       |Description                  |
|-------------------|-------------------|-----------------------------|
|/lm_text_response  |std_msgs/String    |AI responses (JSON formatted)|
|/lm_studio_status  |std_msgs/String    |Node status and health       |
|/available_models  |std_msgs/String    |List of available models     |

# Examples
## 1. Basic Text Chat
```bash
# Send a text message
ros2 topic pub /chat_input std_msgs/String "data: 'Hello, how are you?'"
# Response will be published to /lm_text_response
ros2 topic echo /lm_text_response
```

## 2. Image Analysis
```bash
# Method A: Send image file path
ros2 topic pub /image_file_input std_msgs/String "data: '/home/user/image.jpg'"

# Method B: Send ROS Image message (from camera)
# (Typically published by other nodes like usb_cam)

# Then ask about the image
ros2 topic pub /chat_with_image std_msgs/String "data: 'Describe what you see in this image'"
```

## 3. Model Management
``` bash
# List available models
ros2 topic echo /available_models

# Switch to a different model
ros2 topic pub /set_model std_msgs/String "data: 'qwen2-vl-2b-instruct'"

# Check status
ros2 topic echo /lm_studio_status
```

## 4. Text Completion
``` bash
# Generate text completion
ros2 topic pub /completion_input std_msgs/String "data: 'The future of robotics is'"
```

## 5. Reset Conversation
``` bash
# Clear chat history
ros2 topic pub /reset_chat std_msgs/String "data: 'reset'"
```
## - Response Format
All responses are published as JSON strings on /lm_text_response:

```json
{
  "text": "The generated response text",
  "type": "chat_with_image",
  "model": "qwen2-vl-2b-instruct",
  "timestamp": 1700000000,
  "history_length": 3
}
```
## - Status Messages
Status updates are published on /lm_studio_status:

```json
{
  "model": "qwen2-vl-2b-instruct",
  "chat_history_length": 5,
  "available_models": 3,
  "health_status": true
}
```

## - Available Models
The node automatically discovers and publishes available models:

```json
{
  "models": ["model1", "model2", "qwen2-vl-2b-instruct"],
  "count": 3,
  "current_model": "qwen2-vl-2b-instruct"
}
```

## - Error Handling
The node includes robust error handling:

- Falls back to mock responses when LM Studio is unavailable

- Provides detailed error messages in responses

- Maintains operation during temporary connectivity issues

- Logs all errors to ROS2 logger

## - Integration Examples
- With USB Camera
```bash
# Terminal 1: Start camera
ros2 run usb_cam usb_cam_node_exe

# Terminal 2: Start LM Studio node
ros2 run lm_studio_node lm_studio_node

# Terminal 3: Process camera images
ros2 topic pub /image_input sensor_msgs/Image <camera_topic>
```
- Python Client Example
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class LMStudioClient(Node):
    def __init__(self):
        super().__init__('lm_studio_client')
        self.publisher = self.create_publisher(String, '/chat_input', 10)
        self.subscription = self.create_subscription(
            String,
            '/lm_text_response',
            self.response_callback,
            10
        )
    
    def send_message(self, text):
        msg = String()
        msg.data = text
        self.publisher.publish(msg)
    
    def response_callback(self, msg):
        response = json.loads(msg.data)
        print(f"AI: {response['text']}")
```

# Usage
```python
rclpy.init()
client = LMStudioClient()
client.send_message("Hello, robot!")
rclpy.spin(client)
```

# Troubleshooting
Common Issues
- LM Studio not running:

```bash
# Start LM Studio first
# Check connection:
curl http://localhost:1234/v1/models
```

- No VLM model loaded:

```bash
# Download a vision model
lms get qwen2-vl-2b-instruct
```

- Image format issues:

Supported formats: JPEG, PNG, WebP

Use image_file_input for file paths

Use image_input for ROS Image messages

- Model not responding:

```bash
# Check available models
ros2 topic echo /available_models

# Switch model if needed
ros2 topic pub /set_model std_msgs/String "data: 'different-model'"
```
- Debug Mode
Enable debug logging to see detailed API interactions:

```bash
ros2 run lm_studio_node lm_studio_node --ros-args --log-level debug
```

# Performance Tips
- Use appropriate model sizes for your hardware

- Adjust max_tokens based on response length needs

- Set temperature lower (0.1-0.3) for more deterministic responses

- Use reset_chat periodically to manage context length

- Monitor /lm_studio_status for system health

- License
This project is licensed under the MIT License - see the LICENSE file for details.

# Support
For issues and questions:

- Check LM Studio documentation: lmstudio.ai/docs

- Ensure LM Studio is running and accessible

- Verify you have compatible models downloaded

- Check ROS2 topic connections with ros2 topic list

# Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
