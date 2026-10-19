import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='javier',
    maintainer_email='javifc67@gmail.com',
    description='Paquete de deteccion de obstaculos y gestion de seguridad con LIDAR y camara para ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_obstacle_detector = obstacle_detector.lidar_obstacle_detector:main',
            'camera_obstacle_detector = obstacle_detector.camera_obstacle_detector:main',
            'obstacle_manager = obstacle_detector.obstacle_manager:main',
        ],
    },
)
