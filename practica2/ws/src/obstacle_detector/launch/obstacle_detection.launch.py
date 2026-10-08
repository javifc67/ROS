#!/usr/bin/env python3
"""
Lanzador conjunto (Launch file) para el sistema de deteccion y fusion de obstaculos.
Arranca los nodos de LIDAR, Camara y el Gestor de Seguridad con paso de argumentos.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Declaracion de argumentos de lanzamiento configurables por terminal
    warning_distance_arg = DeclareLaunchArgument(
        'warning_distance',
        default_value='3.0',
        description='Distancia en metros para generar alerta de precaucion'
    )

    danger_distance_arg = DeclareLaunchArgument(
        'danger_distance',
        default_value='1.5',
        description='Distancia en metros para generar alerta de peligro critico'
    )

    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar',
        default_value='true',
        description='Habilitar nodo de deteccion LIDAR'
    )

    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera',
        default_value='true',
        description='Habilitar nodo de deteccion de Camara'
    )

    # Nodos del sistema
    lidar_node = Node(
        package='obstacle_detector',
        executable='lidar_obstacle_detector',
        name='lidar_obstacle_detector',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_lidar')),
        parameters=[{
            'warning_distance': LaunchConfiguration('warning_distance'),
            'danger_distance': LaunchConfiguration('danger_distance'),
            'roi_x_min': 0.2,
            'roi_x_max': 8.0,
            'roi_y_min': -1.5,
            'roi_y_max': 1.5,
        }]
    )

    camera_node = Node(
        package='obstacle_detector',
        executable='camera_obstacle_detector',
        name='camera_obstacle_detector',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_camera')),
        parameters=[{
            'min_contour_area': 3000.0,
            'warning_area': 25000.0,
            'danger_area': 70000.0,
            'publish_annotated_image': True
        }]
    )

    manager_node = Node(
        package='obstacle_detector',
        executable='obstacle_manager',
        name='obstacle_manager',
        output='screen',
        parameters=[{
            'warning_distance': LaunchConfiguration('warning_distance'),
            'emergency_distance': LaunchConfiguration('danger_distance'),
            'fusion_mode': 'pessimistic'
        }]
    )

    return LaunchDescription([
        warning_distance_arg,
        danger_distance_arg,
        enable_lidar_arg,
        enable_camera_arg,
        lidar_node,
        camera_node,
        manager_node
    ])
