#!/usr/bin/env bash
# ==============================================================================
# Script de Demostracion Visual Completa para la Practica 2
# Lanza: Nodos de deteccion + RViz2 (LIDAR 3D) + Visor de Camara + Rosbag en bucle
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR/ws"
BAG_DIR="$SCRIPT_DIR/rosbag2_2025_02_27-13_08_14"
RVIZ_CFG="$SCRIPT_DIR/rviz/practica2_visual.rviz"

# Asegurar entorno grafico para WSLg / X11
export DISPLAY="${DISPLAY:-:0}"

echo "=========================================================="
echo "  PRACTICA 2: DEMOSTRACION VISUAL EN TIEMPO REAL (ROS 2)"
echo "=========================================================="

# Cargar entornos de ROS 2
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null
source "$WS_DIR/install/setup.bash"

# Limpieza ordenada de todos los procesos al salir (Ctrl+C)
PIDS=()
cleanup() {
    echo ""
    echo "=== Deteniendo todos los componentes visuales y nodos ==="
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "=== Sistema detenido correctamente ==="
    exit 0
}
trap cleanup INT TERM EXIT

# 1. Lanzar nodos de deteccion (LIDAR, Camara y Gestor de Seguridad)
echo "[1/4] Lanzando nodos de deteccion y fusion..."
ros2 launch obstacle_detector obstacle_detection.launch.py &
PIDS+=($!)
sleep 2

# 2. Lanzar RViz2 con el perfil visual preconfigurado
echo "[2/4] Abriendo visualizador 3D RViz2 (LIDAR PointCloud)..."
rviz2 -d "$RVIZ_CFG" >/dev/null 2>&1 &
PIDS+=($!)
sleep 2

# 3. Lanzar visor de camara con las anotaciones de deteccion
echo "[3/4] Abriendo ventana de deteccion de Camara..."
ros2 run rqt_image_view rqt_image_view /obstacles/camera/annotated_image >/dev/null 2>&1 &
PIDS+=($!)
sleep 1

# 4. Reproducir el rosbag sensorial en bucle (--loop)
echo "[4/4] Reproduciendo bolsa de datos sensoriales (LIDAR + Camara)..."
echo "----------------------------------------------------------"
echo "  * La reproduccion se ejecuta en bucle (--loop)."
echo "  * Veras la nube de puntos 3D en RViz2 y la camara en rqt."
echo "  * Pulsa Ctrl+C en esta terminal para cerrar todo."
echo "----------------------------------------------------------"

if [ -d "$BAG_DIR" ]; then
    ros2 bag play "$BAG_DIR" \
        --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw /tf_static \
        --rate 1.0 \
        --loop
else
    echo "ERROR: No se encuentra la carpeta del rosbag en $BAG_DIR"
    wait
fi
