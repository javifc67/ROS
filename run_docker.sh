#!/usr/bin/env bash

# Permitir que el contenedor use el servidor gráfico X11 local
xhost +local:root > /dev/null 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Construir la imagen si no existe
if ! docker image inspect ros2_jazzy_turtlesim > /dev/null 2>&1; then
    echo "=== Construyendo imagen Docker ros2_jazzy_turtlesim ==="
    docker build -t ros2_jazzy_turtlesim -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
fi

echo "=== Iniciando contenedor ROS 2 Jazzy con interfaz gráfica ==="
docker run -it --rm \
    --name ros2_turtlesim_container \
    --net=host \
    --ipc=host \
    --privileged \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$SCRIPT_DIR:/workspace" \
    ros2_jazzy_turtlesim \
    bash
