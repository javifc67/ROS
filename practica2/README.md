# Práctica 2: Detección y Fusión Sensorial de Obstáculos (ROS 2)

Este paquete implementa un pipeline completo de percepción para vehículos autónomos/robótica móvil en ROS 2 (probado en Jazzy y Humble), integrando detección con **LIDAR 3D** (Point Cloud), **Cámara monocromática** (Visión artificial con OpenCV) y un **Gestor de Seguridad** con fusión sensorial y servicio síncrono.

---

## ⚠️ Nota Importante sobre los Datos Sensoriales (Rosbag de ~36 GB)

El conjunto de datos sensoriales (`rosbag2_2025_02_27-13_08_14`, ~36 GB) **no está incluido en el repositorio Git** por restricciones de tamaño de GitHub.

Para reproducir la prueba con el rosbag:
1. Copia o descarga la carpeta del bag en:
   ```bash
   practica2/rosbag2_2025_02_27-13_08_14/
   ```
2. O bien reprodúcelo desde cualquier otra ruta especificando los tópicos necesarios:
   ```bash
   ros2 bag play <ruta_al_rosbag> --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw --rate 1.0
   ```

---

## Estructura del Proyecto

```text
practica2/
├── run_demo.sh                  # Script de arranque automático (compila, lanza nodos y reproduce bag si existe)
├── Memoria_Practica2.pdf        # Memoria explicativa completa del diseño y resultados
├── Memoria_Practica2.html       # Versión web responsive e interactiva de la memoria
├── ws/                          # Espacio de trabajo de ROS 2
│   └── src/
│       ├── obstacle_detector_interfaces/  # Mensajes (ObstacleInfo.msg) y servicios (QuerySafetyStatus.srv)
│       └── obstacle_detector/             # Nodos de detección LIDAR, Cámara y Gestor de Fusión
└── README.md
```

---

## Requisitos y Dependencias

- **ROS 2** (Jazzy Jalisco o Humble Hawksbill).
- Paquetes de ROS 2:
  - `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, `cv_bridge`
- Python:
  - `numpy`, `opencv-python` (cv2)

> **Nota:** Si no tienes ROS 2 instalado en el sistema host, puedes ejecutar el contenedor Docker provisto en la raíz del repositorio (`./run_docker.sh`).

---

## Compilación y Ejecución

### Opción 1: Ejecución Rápida Automatizada
Desde la raíz del repositorio o dentro de `practica2/`:
```bash
cd practica2
chmod +x run_demo.sh
./run_demo.sh
```
Este script:
1. Detecta si estás en el host o en Docker.
2. Compila el workspace si no está compilado aún (`colcon build`).
3. Hace `source` del entorno.
4. Lanza los tres nodos (`ros2 launch obstacle_detector obstacle_detection.launch.py`).
5. Si detecta la carpeta del rosbag, comienza la reproducción automáticamente.

---

### Opción 2: Compilación y Lanzamiento Manual

1. **Compilar el espacio de trabajo:**
   ```bash
   cd practica2/ws
   colcon build
   source install/setup.bash
   ```

2. **Lanzar los nodos de detección:**
   ```bash
   ros2 launch obstacle_detector obstacle_detection.launch.py
   ```

3. **Reproducir el rosbag (en otra terminal):**
   ```bash
   ros2 bag play practica2/rosbag2_2025_02_27-13_08_14 --topics /ouster/points /my_camera/pylon_ros2_camera_node/image_raw --rate 1.0
   ```

---

## Pruebas y Comandos de Inspección

### 1. Consultar el estado de seguridad (Servicio ROS 2)
En una terminal con el entorno cargado (`source practica2/ws/install/setup.bash`):

- **Consultar todos los sensores:**
  ```bash
  ros2 service call /safety/query_status obstacle_detector_interfaces/srv/QuerySafetyStatus "{sensor_filter: 'all'}"
  ```
- **Filtrar solo por LIDAR o Cámara:**
  ```bash
  ros2 service call /safety/query_status obstacle_detector_interfaces/srv/QuerySafetyStatus "{sensor_filter: 'lidar'}"
  ros2 service call /safety/query_status obstacle_detector_interfaces/srv/QuerySafetyStatus "{sensor_filter: 'camera'}"
  ```

### 2. Modificación de Parámetros en Tiempo Real
- **Ajustar la distancia de advertencia:**
  ```bash
  ros2 param set /obstacle_manager warning_distance 4.5
  ```
- **Ajustar la distancia de parada de emergencia:**
  ```bash
  ros2 param set /obstacle_manager stop_distance 1.8
  ```

### 3. Visualizar imágenes anotadas de la cámara
```bash
ros2 run rqt_image_view rqt_image_view
```
Seleccionar el tópico `/camera/obstacle_image` para ver los cuadros delimitadores (bounding boxes) y las estimaciones de distancia en tiempo real.
