from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declaración de argumentos del launch
    speed_arg = DeclareLaunchArgument(
        'speed',
        default_value='2.0',
        description='Velocidad lineal de la tortuga'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='2.0',
        description='Velocidad angular de la tortuga'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Nivel de logging (DEBUG, INFO, WARN, ERROR)'
    )

    # Nodo de Turtlesim
    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        output='screen'
    )

    # Nodo de control por teclado con paso de parámetros
    teleop_node = Node(
        package='turtle_control',
        executable='teleop_turtle',
        name='teleop_turtle',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'speed': LaunchConfiguration('speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'log_level': LaunchConfiguration('log_level'),
        }]
    )

    return LaunchDescription([
        speed_arg,
        angular_speed_arg,
        log_level_arg,
        turtlesim_node,
        teleop_node,
    ])
