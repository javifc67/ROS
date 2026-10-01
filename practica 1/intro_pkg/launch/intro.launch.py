from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='intro_pkg',
            executable='pub_node',
            name='pub_alternativo'
        ),
        Node(
            package='intro_pkg',
            executable='pub_node',
        ),
        Node(
            package='intro_pkg',
            executable='sub_node',
        ),
    ])
