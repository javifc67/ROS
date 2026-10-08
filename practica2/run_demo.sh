#!/usr/bin/env bash
# Script de ejecucion y demostracion rapida para la Practica 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR/ws"
BAG_DIR="$SCRIPT_DIR/rosbag2_2025_02_27-13_08_14"

# Si ROS 2 no esta instalado de forma nativa en el host, ejecutar transparentemente dentro del contenedor Docker
if ! command -v ros2 >/dev/null 2>&1; then
    echo "=========================================================="
    echo "  ROS 2 no detectado en el host."
    echo "  Ejecutando la Practica 2 dentro del entorno Docker..."
    echo "=========================================================="
    xhost +local:root > /dev/null 2>&1 || true
    ROS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    exec docker run -it --rm \
        --net=host \
        --ipc=host \
        --privileged \
        -e DISPLAY="${DISPLAY:-:0}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$ROS_ROOT:/workspace" \
        ros2_jazzy_turtlesim \
        bash -c "cd /workspace/practica2 && ./run_demo.sh"
fi

echo "=========================================================="
echo "  PRACTICA 2: DETECCION Y FUSION DE OBSTACULOS EN ROS 2"
echo "=========================================================="

source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null || true

if [ ! -d "$WS_DIR/install" ]; then
    echo "=== Compilando el espacio de trabajo con colcon build ==="
    cd "$WS_DIR"
    colcon build
fi

source "$WS_DIR/install/setup.bash"

echo "=== Lanzando nodos de deteccion (LIDAR, Camara y Gestor) ==="
ros2 launch obstacle_detector obstacle_detection.launch.py &
LAUNCH_PID=$!

trap "echo 'Deteniendo procesos...'; kill $LAUNCH_PID 2>/dev/null || true; exit" INT TERM EXIT

sleep 3

echo ""
echo "=========================================================="
echo "  SISTEMA ACTIVO: Reproduciendo bolsa sensorial (ROS 2 Bag)"
echo "=========================================================="
echo "Puedes llamar al servicio en otra terminal con:"
echo '  ros2 service call /safety/query_status obstacle_detector_interfaces/srv/QuerySafetyStatus "{sensor_filter: '\''all'\''}"'
echo "Modificar parametros en caliente con:"
echo '  ros2 param set /obstacle_manager warning_distance 4.5'
echo "=========================================================="
echo ""

if [ -d "$BAG_DIR" ]; then
    ros2 bag play "$BAG_DIR" --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw --rate 1.0
else
    echo "=========================================================="
    echo "  AVISO: No se ha encontrado la carpeta del rosbag ($BAG_DIR)"
    echo "  (No incluida en Git por su gran tamaño, ~36 GB)."
    echo ""
    echo "  Si dispones del rosbag de la práctica, colócalo en:"
    echo "    $BAG_DIR"
    echo "  O reprodúcelo manualmente con:"
    echo "    ros2 bag play <ruta_bag> --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw"
    echo ""
    echo "  Los nodos de detección ya están activos y listos."
    echo "  Pulsa Ctrl+C para finalizar la ejecución."
    echo "=========================================================="
    wait $LAUNCH_PID
fi
