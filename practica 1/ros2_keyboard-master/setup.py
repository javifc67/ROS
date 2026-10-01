from setuptools import setup
from glob import glob

package_name = 'ros2_keyboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/{}'.format(package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'PyYAML', 'pynput'],
    zip_safe=True,
    maintainer='Alfredo Valle',
    maintainer_email='alfredo.valle@upm.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_node = ros2_keyboard.keyboard_node:main'
        ],
    },
)
