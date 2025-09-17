from setuptools import setup
import os
from glob import glob

package_name = 'lm_studio_connector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nadim Arubai',
    maintainer_email='nadim.arubai@gmail.com',
    description='ROS2 node for LM Studio API communication',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lm_studio_node = lm_studio_connector.lm_studio_node:main',
            'lm_studio_topics_node = lm_studio_connector.lm_studio_topics_node:main',
        ],
    },
)
