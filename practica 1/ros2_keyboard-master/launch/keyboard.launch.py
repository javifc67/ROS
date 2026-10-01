from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
  parameters_file_path = '{}/conf/'.format(Path(__file__).parents[1])

  return LaunchDescription([
    Node(
      package='ros2_keyboard',
      node_executable='keyboard_node',
      arguments=[parameters_file_path+'keyboard.yaml'],
      output='screen',
      emulate_tty=True
    )
  ])
