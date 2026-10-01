FROM osrf/ros:jazzy-desktop

# Instalar dependencias adicionales para teclado y herramientas útiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-pynput \
    ros-jazzy-turtlesim \
    ros-jazzy-example-interfaces \
    x11-apps \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Configurar entorno de ROS automáticamente en bash
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]
