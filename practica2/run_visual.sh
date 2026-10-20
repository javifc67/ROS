#!/usr/bin/env bash
# ==============================================================================
# Script de Demostracion Visual Completa para la Practica 2
# Lanza: Nodos de deteccion + RViz2 (LIDAR 3D) + Ventana Camara YOLO + Rosbag Loop
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

# 1. Lanzar bolsa sensorial en segundo plano (para que los datos ya esten fluyendo)
echo "[1/3] Iniciando reproduccion de bolsa sensorial (LIDAR + Camara)..."
ros2 bag play "$BAG_DIR" \
    --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw /tf_static \
    --rate 1.0 \
    --loop >/dev/null 2>&1 &
PIDS+=($!)
sleep 2

# 2. Lanzar nodos de deteccion (LIDAR, Camara con YOLO y Gestor)
echo "[2/3] Lanzando nodos de deteccion y abriendo ventana de camara..."
ros2 launch obstacle_detector obstacle_detection.launch.py &
PIDS+=($!)
sleep 2

# 3. Lanzar RViz2 con el perfil 3D preconfigurado
echo "[3/3] Abriendo visualizador 3D RViz2 (LIDAR PointCloud)..."
rviz2 -d "$RVIZ_CFG" >/dev/null 2>&1 &
PIDS+=($!)

echo "----------------------------------------------------------"
echo "  * La reproduccion se ejecuta en bucle continuo (--loop)."
echo "  * Se abrira la ventana interactiva 'Detección Cámara' con YOLO."
echo "  * En RViz2 veras la nube de puntos 3D."
echo "  * Pulsa Ctrl+C en esta terminal para cerrar todo."
echo "----------------------------------------------------------"

wait
