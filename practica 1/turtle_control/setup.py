import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'turtle_control'

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
    maintainer='Alumno Robotica',
    maintainer_email='robotica@etsisi.upm.es',
    description='Control avanzado para Turtlesim con movimiento fluido y servicios',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_turtle = turtle_control.teleop_turtle:main',
        ],
    },
)
