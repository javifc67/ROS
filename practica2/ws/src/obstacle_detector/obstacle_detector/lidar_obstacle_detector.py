#!/usr/bin/env python3
"""
Nodo de deteccion de obstaculos mediante LIDAR Ouster (PointCloud2).
Procesa la nube de puntos 3D en tiempo real, aplica filtrado espacial ROI,
detecta obstaculos proximos, calcula distancias euclideas y publica alertas
en /obstacles/lidar, marcadores 3D en /obstacles/lidar/markers y la nube
filtrada del obstaculo en /obstacles/lidar/roi_points para RViz2.
Soporta reconfiguracion de parametros en tiempo de ejecucion.
"""

import math
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

from obstacle_detector_interfaces.msg import ObstacleInfo


class LidarObstacleDetector(Node):
    """
    Nodo que procesa nubes de puntos de LIDAR para deteccion frontal de obstaculos
    y generacion de marcas visuales 3D para RViz2.
    """

    def __init__(self):
        super().__init__('lidar_obstacle_detector')

        # Declaracion de parametros configurables en tiempo de ejecucion
        self.declare_parameter('roi_x_min', 0.2)
        self.declare_parameter('roi_x_max', 8.0)
        self.declare_parameter('roi_y_min', -1.5)
        self.declare_parameter('roi_y_max', 1.5)
        self.declare_parameter('roi_z_min', -1.0)
        self.declare_parameter('roi_z_max', 1.5)
        self.declare_parameter('warning_distance', 3.0)
        self.declare_parameter('danger_distance', 1.5)
        self.declare_parameter('min_points_cluster', 15)

        # Cargar valores iniciales
        self._load_parameters()

        # Callback dinamico para actualizacion de parametros en tiempo de ejecucion
        self.add_on_set_parameters_callback(self._on_parameters_change)

        # Suscriptor al topic de LIDAR del bag (/ouster/points)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/ouster/points',
            self._pointcloud_callback,
            qos_profile_sensor_data
        )

        # Publicador de informacion de obstaculos (mensaje custom)
        self.publisher = self.create_publisher(
            ObstacleInfo,
            '/obstacles/lidar',
            10
        )

        # Publicador de marcadores visuales 3D para RViz2 (Bounding Box, Esfera y Texto)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/obstacles/lidar/markers',
            10
        )

        # Publicador de nube de puntos filtrada (solo puntos del obstaculo detectado en el pasillo)
        self.roi_cloud_pub = self.create_publisher(
            PointCloud2,
            '/obstacles/lidar/roi_points',
            10
        )

        self.get_logger().info(
            'LidarObstacleDetector inicializado con soporte de visualizacion 3D RViz. '
            f'ROI X: [{self.roi_x_min:.1f}, {self.roi_x_max:.1f}]m, '
            f'Distancia Peligro: {self.danger_distance:.1f}m, '
            f'Alerta: {self.warning_distance:.1f}m'
        )

    def _load_parameters(self):
        """Carga y almacena los parametros del nodo."""
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.roi_z_min = float(self.get_parameter('roi_z_min').value)
        self.roi_z_max = float(self.get_parameter('roi_z_max').value)
        self.warning_distance = float(self.get_parameter('warning_distance').value)
        self.danger_distance = float(self.get_parameter('danger_distance').value)
        self.min_points_cluster = int(self.get_parameter('min_points_cluster').value)

    def _on_parameters_change(self, params):
        """Callback ejecutado cuando se modifica un parametro dinamicamente con ros2 param set."""
        for param in params:
            self.get_logger().info(f'Actualizando parametro dinámico de LIDAR "{param.name}" -> {param.value}')
            if param.name == 'roi_x_min':
                self.roi_x_min = float(param.value)
            elif param.name == 'roi_x_max':
                self.roi_x_max = float(param.value)
            elif param.name == 'roi_y_min':
                self.roi_y_min = float(param.value)
            elif param.name == 'roi_y_max':
                self.roi_y_max = float(param.value)
            elif param.name == 'roi_z_min':
                self.roi_z_min = float(param.value)
            elif param.name == 'roi_z_max':
                self.roi_z_max = float(param.value)
            elif param.name == 'warning_distance':
                self.warning_distance = float(param.value)
            elif param.name == 'danger_distance':
                self.danger_distance = float(param.value)
            elif param.name == 'min_points_cluster':
                self.min_points_cluster = int(param.value)

        return SetParametersResult(successful=True)

    def _pointcloud_callback(self, msg: PointCloud2):
        """Procesa cada mensaje de PointCloud2 del LIDAR."""
        try:
            # Extraccion eficiente de coordenadas XYZ usando numpy
            struct_arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=False)
            if struct_arr.size == 0:
                return
            pts = np.column_stack((struct_arr['x'], struct_arr['y'], struct_arr['z']))
            if pts.size == 0:
                return

            # Mascara de puntos finitos
            valid = np.isfinite(pts).all(axis=1)
            pts_finite = pts[valid]

            # Transformacion a marco frontal de avance del vehiculo (os_sensor):
            # En el Ouster LiDAR, os_lidar tiene una rotacion de 180 deg en yaw respecto a os_sensor.
            # En os_sensor: +X es hacia adelante (avance), +Y es hacia la izquierda, +Z hacia arriba.
            pts_sensor = np.empty_like(pts_finite)
            pts_sensor[:, 0] = -pts_finite[:, 0]
            pts_sensor[:, 1] = -pts_finite[:, 1]
            pts_sensor[:, 2] = pts_finite[:, 2] - 0.038

            mask_roi = (
                (pts_sensor[:, 0] >= self.roi_x_min) &
                (pts_sensor[:, 0] <= self.roi_x_max) &
                (pts_sensor[:, 1] >= self.roi_y_min) &
                (pts_sensor[:, 1] <= self.roi_y_max) &
                (pts_sensor[:, 2] >= self.roi_z_min) &
                (pts_sensor[:, 2] <= self.roi_z_max)
            )

            roi_points = pts_sensor[mask_roi]

            # Header referenciado a os_sensor (marco frontal del vehiculo)
            sensor_header = msg.header
            sensor_header.frame_id = 'os_sensor'

            # Construccion del mensaje custom ObstacleInfo
            out_msg = ObstacleInfo()
            out_msg.header = sensor_header
            out_msg.sensor_type = 'lidar'

            closest_point = None
            min_dist = float('inf')
            detected = False
            severity = 'SAFE'

            if len(roi_points) >= self.min_points_cluster:
                detected = True
                # Distancia euclidea en el plano horizontal (XY)
                dists = np.linalg.norm(roi_points[:, :2], axis=1)
                min_idx = np.argmin(dists)
                closest_point = roi_points[min_idx]
                min_dist = float(dists[min_idx])

                out_msg.obstacle_detected = True
                out_msg.distance = min_dist
                out_msg.x = float(closest_point[0])
                out_msg.y = float(closest_point[1])
                out_msg.z = float(closest_point[2])

                if min_dist <= self.danger_distance:
                    severity = 'DANGER'
                    self.get_logger().error(
                        f'[LIDAR-PELIGRO] Obstaculo a {min_dist:.2f}m en (X={out_msg.x:.2f}, Y={out_msg.y:.2f})',
                        throttle_duration_sec=1.0
                    )
                elif min_dist <= self.warning_distance:
                    severity = 'WARNING'
                    self.get_logger().warn(
                        f'[LIDAR-AVISO] Obstaculo a {min_dist:.2f}m en (X={out_msg.x:.2f}, Y={out_msg.y:.2f})',
                        throttle_duration_sec=1.0
                    )
                else:
                    severity = 'SAFE'

                out_msg.severity = severity
            else:
                out_msg.obstacle_detected = False
                out_msg.distance = float('inf')
                out_msg.x = 0.0
                out_msg.y = 0.0
                out_msg.z = 0.0
                out_msg.severity = 'SAFE'

            # 1. Publicar estado estructurado de obstaculo
            self.publisher.publish(out_msg)

            # 2. Publicar nube de puntos filtrada en el pasillo para RViz2
            if detected and len(roi_points) > 0:
                cloud_msg = pc2.create_cloud_xyz32(sensor_header, roi_points)
                self.roi_cloud_pub.publish(cloud_msg)

            # 3. Publicar marcadores 3D interactivos para RViz2
            self._publish_visual_markers(sensor_header, detected, closest_point, min_dist, severity)

        except Exception as e:
            self.get_logger().error(f'Error procesando PointCloud2: {str(e)}', throttle_duration_sec=2.0)

    def _publish_visual_markers(self, header, detected, closest_point, min_dist, severity):
        """Genera y publica marcadores 3D para el visor RViz2."""
        marker_array = MarkerArray()

        # 1. Caja semitransparente que representa el pasillo de avance (ROI)
        box_marker = Marker()
        box_marker.header = header
        box_marker.ns = 'lidar_roi_corridor'
        box_marker.id = 0
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD

        # Centro geometrico de la caja ROI
        box_marker.pose.position.x = (self.roi_x_min + self.roi_x_max) / 2.0
        box_marker.pose.position.y = (self.roi_y_min + self.roi_y_max) / 2.0
        box_marker.pose.position.z = (self.roi_z_min + self.roi_z_max) / 2.0
        box_marker.pose.orientation.w = 1.0

        # Dimensiones de la caja
        box_marker.scale.x = max(0.1, float(self.roi_x_max - self.roi_x_min))
        box_marker.scale.y = max(0.1, float(self.roi_y_max - self.roi_y_min))
        box_marker.scale.z = max(0.1, float(self.roi_z_max - self.roi_z_min))

        # Color dinamico segun severidad
        if severity == 'DANGER':
            box_marker.color.r = 1.0
            box_marker.color.g = 0.1
            box_marker.color.b = 0.1
            box_marker.color.a = 0.35
        elif severity == 'WARNING':
            box_marker.color.r = 1.0
            box_marker.color.g = 0.75
            box_marker.color.b = 0.0
            box_marker.color.a = 0.28
        else: # SAFE
            box_marker.color.r = 0.0
            box_marker.color.g = 0.85
            box_marker.color.b = 0.2
            box_marker.color.a = 0.18

        marker_array.markers.append(box_marker)

        # 2. Esfera en la posicion exacta del obstaculo mas cercano
        sphere_marker = Marker()
        sphere_marker.header = header
        sphere_marker.ns = 'lidar_closest_obstacle'
        sphere_marker.id = 1
        sphere_marker.type = Marker.SPHERE

        # 3. Texto 3D flotante indicando la distancia metrica y severidad
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = 'lidar_obstacle_text'
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING

        if detected and closest_point is not None:
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(closest_point[0])
            sphere_marker.pose.position.y = float(closest_point[1])
            sphere_marker.pose.position.z = float(closest_point[2])
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.45
            sphere_marker.scale.y = 0.45
            sphere_marker.scale.z = 0.45

            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(closest_point[0])
            text_marker.pose.position.y = float(closest_point[1])
            text_marker.pose.position.z = float(closest_point[2]) + 0.65
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.45
            text_marker.text = f"LIDAR: {min_dist:.2f}m [{severity}]"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            if severity == 'DANGER':
                sphere_marker.color.r = 1.0
                sphere_marker.color.g = 0.0
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 0.95
            elif severity == 'WARNING':
                sphere_marker.color.r = 1.0
                sphere_marker.color.g = 0.6
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 0.90
            else:
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 0.8
                sphere_marker.color.b = 0.2
                sphere_marker.color.a = 0.85
        else:
            sphere_marker.action = Marker.DELETE
            text_marker.action = Marker.DELETE

        marker_array.markers.append(sphere_marker)
        marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
