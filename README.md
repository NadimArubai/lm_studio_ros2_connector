# LM Studio ROS2 Connector

A ROS2 node for integrating LM Studio's local LLM API with robotics applications, featuring both Action/Service interface for robust integration and Topic interface for debugging/backward compatibility.

## Features

🤖 **Dual Interface**: Action/Service for production, Topics for debugging  
🖼️ **Multimodal Support**: Image input processing for Vision-Language Models (VLMs)  
📝 **Multiple Modalities**: Chat completion, text completion, and embeddings  
🔍 **Model Management**: Dynamic model switching and discovery  
⚡ **Real-time**: ROS2 interfaces for seamless robotics integration  
🏥 **Health Monitoring**: Connection status and health checks  
🎯 **Production Ready**: Action servers for reliable long-running tasks  
🚀 **Streaming Support**: Real-time token streaming with progress feedback  

## Prerequisites

- ROS 2 (Tested on Jazyy, but should work with Foxy, Galactic, Humble, or newer)
- Python (Tested on 3.12, but should work with 3.8+)
- LM Studio running locally (default: http://localhost:1234)
- A Vision-Language Model (VLM) like qwen2-vl-2b-instruct for image support

## Installation

```bash
# Install LM Studio from lmstudio.ai

# Download a VLM (for image support)
lms get qwen2-vl-2b-instruct

# Add to your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/NadimArubai/lm_studio_ros2_connector.git
cd ..
colcon build --packages-select lm_studio_connector lm_studio_interfaces
source install/setup.bash
```

## Node Types

### 1. Main Node (Actions/Services - Recommended)
```bash
ros2 run lm_studio_connector lm_studio_node
```
**Production-ready** with action servers and services for reliable integration.

### 2. Debug Node (Topics - Legacy)
```bash
ros2 run lm_studio_connector lm_studio_topics_node
```
**Backward compatibility** with the old topic-based interface.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lm_studio_url` | `http://localhost:1234` | LM Studio API URL |
| `api_key` | `""` | API key (if required) |
| `model_name` | `"local-model"` | Default model to use |
| `max_tokens` | `500` | Maximum response tokens |
| `temperature` | `0.7` | Creativity level (0.0-1.0) |
| `timeout` | `30` | API timeout in seconds |
| `stream` | `false` | Stream responses |
| `max_history_length` | `10` | Chat history context length |

## Action Interface (Recommended)

### Chat Completion Action
**Action Name**: `/chat_completion`

```python
# Request
string prompt
string model
int32 max_tokens
float32 temperature
float32 timeout
bool use_history
string image_data       # "latest" or image path or base64
bool stream             # Enable real-time token streaming
bool progress_feedback  # Enable periodic progress updates

# Result  
string response
bool success

# Feedback
string partial_response
string status
float32 progress
```

**Streaming Modes:**
- `stream=true`: Real-time token-by-token feedback
- `progress_feedback=true`: Periodic progress updates during processing
- Both false: Standard blocking request

image_reference could be:
* 'latest' for last sending image via the image topic or filepath topic.
* 'image_path' for a saved image on the hard.
* a string start with "data:image/" for base64.

### Text Completion Action
**Action Name**: `/text_completion`

```python
# Request
string prompt
string model
int32 max_tokens
float32 temperature
float32 timeout
bool stream             # Enable real-time token streaming
bool progress_feedback  # Enable periodic progress updates

# Result
string response
bool success

# Feedback
string partial_response
string status
float32 progress
```

## Service Interface

### Available Services:
- `/list_models` - Get available models
- `/get_embeddings` - Generate text embeddings  
- `/set_model` - Change active model
- `/reset_chat` - Clear conversation history

## Topic Interface (Legacy/Debug)

### Subscribers (Input)
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/chat_input` | `std_msgs/String` | Text chat messages |
| `/completion_input` | `std_msgs/String` | Text completion prompts |
| `/image_input` | `sensor_msgs/Image` | Raw image data |
| `/image_input/compressed` | `sensor_msgs/CompressedImage` | Compressed image data |
| `/image_file_input` | `std_msgs/String` | Path to image file |
| `/chat_with_image` | `std_msgs/String` | Text prompt for current image |

### Publishers (Output)
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/lm_text_response` | `std_msgs/String` | AI responses (JSON formatted) |
| `/lm_studio_status` | `std_msgs/String` | Node status and health |
| `/available_models` | `std_msgs/String` | List of available models |

## Usage Examples

### 1. Using Actions (Recommended)

#### Standard Chat (Blocking)
```bash
# Chat with image (using latest received image)
ros2 action send_goal /chat_completion lm_studio_interfaces/action/ChatCompletion "
prompt: 'Describe this image'
use_history: true
image_reference: 'latest'
max_tokens: 300
temperature: 0.3
stream: false
progress_feedback: false
"
```

#### Streaming Chat (Real-time tokens)
```bash
# Stream response tokens in real-time
ros2 action send_goal /chat_completion lm_studio_interfaces/action/ChatCompletion "
prompt: 'Explain the future of robotics'
use_history: true
max_tokens: 500
stream: true
progress_feedback: false
" --feedback
```

#### Progress Feedback Chat
```bash
# Get periodic progress updates during long processing
ros2 action send_goal /chat_completion lm_studio_interfaces/action/ChatCompletion "
prompt: 'Write a detailed analysis'
max_tokens: 1000
stream: false
progress_feedback: true
" --feedback
```

#### Text Completion with Streaming
```bash
# Text completion with real-time streaming
ros2 action send_goal /text_completion lm_studio_interfaces/action/TextCompletion "
prompt: 'The future of robotics is'
max_tokens: 200
stream: true
progress_feedback: false
" --feedback
```

### 2. Using Services
```bash
# List available models
ros2 service call /list_models lm_studio_interfaces/srv/ListModels

# Get embeddings
ros2 service call /get_embeddings lm_studio_interfaces/srv/GetEmbeddings "text: 'Hello world'"

# Set model
ros2 service call /set_model lm_studio_interfaces/srv/SetModel "model_name: 'qwen2-vl-2b-instruct'"

# Reset chat
ros2 service call /reset_chat lm_studio_interfaces/srv/ResetChat
```

### 3. Using Topics (Legacy)
```bash
# Send image
ros2 topic pub /image_file_input std_msgs/String "data: '/path/to/image.jpg'"

# Chat with image
ros2 topic pub /chat_with_image std_msgs/String "data: 'Describe this image'"

# Monitor responses
ros2 topic echo /lm_text_response
```

## Image Handling

### Supported Methods:
1. **ROS2 Image Messages** (`sensor_msgs/Image`) - Real-time camera data
2. **File Paths** - Pre-captured images
3. **Compressed Images** - Efficient transport
4. **Image Data** - Reference previously sent images

### Best Practices:
```bash
# First run/launch the stream action/service node
ros2 launch lm_studio_connector lm_studio.launch.py

# Send image first
ros2 topic pub /image_file_input std_msgs/String "data: '/home/user/image.jpg'"

# Then reference it in actions
ros2 action send_goal /chat_completion ChatCompletion "
prompt: 'Describe this image'
image_data: 'latest'
stream: true
" --feedback
```

## Response Format

**JSON responses on `/lm_text_response`:**
```json
{
  "text": "The generated response text",
  "type": "chat_action_stream", 
  "model": "qwen2-vl-2b-instruct",
  "timestamp": 1700000000,
  "history_length": 3
}
```

## Python Client Example with Streaming

```python
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
        goal_msg.stream = False
        
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
```

## Integration Examples

### With USB Camera and Streaming
```bash
# Terminal 1: Start camera
ros2 run usb_cam usb_cam_node_exe

# Terminal 2: Start LM Studio node
ros2 run lm_studio_connector lm_studio_node

# Terminal 3: Process images and chat with streaming
ros2 topic pub /image_input sensor_msgs/Image <camera_topic>
ros2 action send_goal /chat_completion ChatCompletion "
prompt: 'What objects are visible?'
image_data: 'latest'
stream: true
" --feedback
```

## Troubleshooting

### Common Issues
**LM Studio not running:**
```bash
# Check connection
curl http://localhost:1234/v1/models
```

**No VLM model loaded:**
```bash
# Download vision model
lms get qwen2-vl-2b-instruct
```

**Streaming not working:**
- Ensure LM Studio supports streaming for your model
- Check that `stream=true` is set in action goal
- Use `--feedback` flag to see streaming output

**Debug mode:**
```bash
ros2 run lm_studio_connector lm_studio_node --ros-args --log-level debug
```

### Performance Tips
- Use appropriate model sizes for your hardware
- Adjust `max_tokens` based on response length needs  
- Set `temperature` lower (0.1-0.3) for deterministic responses
- Use `reset_chat` service to manage context length
- Monitor `/lm_studio_status` for system health
- Use streaming for long responses to provide real-time feedback
- Use progress feedback for very long processing tasks

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check LM Studio documentation: lmstudio.ai/docs
- Ensure LM Studio is running and accessible
- Verify you have compatible models downloaded
- Check ROS2 interface connections
- For streaming issues, verify model supports streaming

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## Maintainer

Nadim Arubai - nadim.arubai@gmail.com
