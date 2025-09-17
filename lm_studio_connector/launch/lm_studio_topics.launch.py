from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lm_studio_connector',
            executable='lm_studio_topics_node',
            name='lm_studio_topics_node',
            parameters=[{
                'ml_studio_url': 'http://localhost:1234',
                'api_key': 'your-api-key-here',
                'model_id': 'gemma-3-4b-it',
                'prediction_timeout': 30
            }],
            output='screen'
        )
    ])
