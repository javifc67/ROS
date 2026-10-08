#!/usr/bin/env python3
"""
Nodo de deteccion de obstaculos mediante LIDAR Ouster (PointCloud2).
Procesa la nube de puntos 3D en tiempo real, aplica filtrado espacial ROI,
detecta obstaculos proximos y publica alertas en el topic /obstacles/lidar.
Soporta reconfiguracion de parametros en tiempo de ejecucion.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult, ParameterType
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from obstacle_detector_interfaces.msg import ObstacleInfo


class LidarObstacleDetector(Node):
    """
    Nodo que procesa nubes de puntos de LIDAR para deteccion frontal de obstaculos.
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
            10
        )

        # Publicador de obstaculos detectados
        self.publisher = self.create_publisher(
            ObstacleInfo,
            '/obstacles/lidar',
            10
        )

        self.get_logger().info(
            'LidarObstacleDetector inicializado correctamente. '
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
            self.get_logger().info(f'Actualizando parametro dinámico "{param.name}" -> {param.value}')
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
            pts = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'])
            if pts.size == 0:
                return

            # Mascara de puntos finitos y filtrado de la Region of Interest (ROI)
            valid = np.isfinite(pts).all(axis=1)
            pts_finite = pts[valid]

            mask_roi = (
                (pts_finite[:, 0] >= self.roi_x_min) &
                (pts_finite[:, 0] <= self.roi_x_max) &
                (pts_finite[:, 1] >= self.roi_y_min) &
                (pts_finite[:, 1] <= self.roi_y_max) &
                (pts_finite[:, 2] >= self.roi_z_min) &
                (pts_finite[:, 2] <= self.roi_z_max)
            )

            roi_points = pts_finite[mask_roi]

            # Construccion del mensaje custom ObstacleInfo
            out_msg = ObstacleInfo()
            out_msg.header = msg.header
            out_msg.sensor_type = 'lidar'

            if len(roi_points) >= self.min_points_cluster:
                # Distancia euclidea en el plano horizontal (XY) o 3D (XYZ)
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
                    out_msg.severity = 'DANGER'
                    self.get_logger().error(
                        f'[LIDAR-PELIGRO] Obstaculo a {min_dist:.2f}m en (X={out_msg.x:.2f}, Y={out_msg.y:.2f})',
                        throttle_duration_sec=1.0
                    )
                elif min_dist <= self.warning_distance:
                    out_msg.severity = 'WARNING'
                    self.get_logger().warn(
                        f'[LIDAR-AVISO] Obstaculo a {min_dist:.2f}m en (X={out_msg.x:.2f}, Y={out_msg.y:.2f})',
                        throttle_duration_sec=1.0
                    )
                else:
                    out_msg.severity = 'SAFE'
            else:
                out_msg.obstacle_detected = False
                out_msg.distance = float('inf')
                out_msg.x = 0.0
                out_msg.y = 0.0
                out_msg.z = 0.0
                out_msg.severity = 'SAFE'

            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error procesando PointCloud2: {str(e)}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
