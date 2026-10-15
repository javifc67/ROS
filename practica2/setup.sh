#!/usr/bin/env bash

# Directorio base de la practica
PRACTICA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$PRACTICA_DIR/ws"

# Comprobar si el script se esta ejecutando con source o directamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  AVISO IMPORTANTE:"
    echo "Para que las variables de entorno de ROS 2 queden activas en tu terminal actual,"
    echo "debes ejecutar este script con 'source' o '.':"
    echo ""
    echo "    source setup.sh"
    echo "    # o bien:"
    echo "    . setup.sh"
    echo ""
    echo "------------------------------------------------------"
fi

# Cargar ROS 2 base (Humble)
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Si no esta compilado aun el workspace, compilarlo con colcon
if [ ! -f "$WS_DIR/install/setup.bash" ]; then
    echo "Compilando el espacio de trabajo con colcon build..."
    (cd "$WS_DIR" && colcon build)
fi

# Cargar el entorno de los paquetes de la Practica 2
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
    echo "✅ Entorno de la Práctica 2 cargado correctamente."
    echo "   Paquetes listos: obstacle_detector, obstacle_detector_interfaces"
    echo ""
    echo "Puedes lanzar el sistema con:"
    echo "   ros2 launch obstacle_detector obstacle_detection.launch.py"
    echo "O ejecutar la demo completa con:"
    echo "   ./run_demo.sh"
else
    echo "❌ Error: No se pudo encontrar $WS_DIR/install/setup.bash"
fi
