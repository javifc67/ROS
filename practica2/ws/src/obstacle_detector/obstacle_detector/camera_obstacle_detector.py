#!/usr/bin/env python3
"""
Nodo de deteccion visual de obstaculos mediante camara RGB (Basler Blaze).
Procesa la transmision de video de la camara, detecta contornos y regiones
de obstaculos frontales mediante vision por computador (OpenCV) y publica alertas
en el topic /obstacles/camera junto a la imagen anotada para visualizacion.
Soporta reconfiguracion de parametros en tiempo de ejecucion.
"""

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from obstacle_detector_interfaces.msg import ObstacleInfo


class CameraObstacleDetector(Node):
    """
    Nodo que procesa imagenes RGB de la camara para deteccion visual de obstaculos.
    """

    def __init__(self):
        super().__init__('camera_obstacle_detector')

        # Parametros configurables dinamicamente
        self.declare_parameter('min_contour_area', 3000.0)
        self.declare_parameter('warning_area', 25000.0)
        self.declare_parameter('danger_area', 70000.0)
        self.declare_parameter('roi_top_ratio', 0.35)
        self.declare_parameter('publish_annotated_image', True)

        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameters_change)

        self.bridge = CvBridge()

        # Suscriptor al topic de camara del bag
        self.subscription = self.create_subscription(
            Image,
            '/my_camera/pylon_ros2_camera_node/image_raw',
            self._image_callback,
            10
        )

        # Publicador de obstaculos de camara
        self.obstacle_pub = self.create_publisher(
            ObstacleInfo,
            '/obstacles/camera',
            10
        )

        # Publicador de imagen con anotaciones graficas para RViz / visualizadores
        self.image_pub = self.create_publisher(
            Image,
            '/obstacles/camera/annotated_image',
            10
        )

        self.get_logger().info(
            'CameraObstacleDetector inicializado. '
            f'Area Minima: {self.min_contour_area:.0f}px, '
            f'Aviso: {self.warning_area:.0f}px, '
            f'Peligro: {self.danger_area:.0f}px'
        )

    def _load_parameters(self):
        """Carga y almacena los parametros del nodo."""
        self.min_contour_area = float(self.get_parameter('min_contour_area').value)
        self.warning_area = float(self.get_parameter('warning_area').value)
        self.danger_area = float(self.get_parameter('danger_area').value)
        self.roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self.publish_annotated_image = bool(self.get_parameter('publish_annotated_image').value)

    def _on_parameters_change(self, params):
        """Callback ejecutado cuando se modifica un parametro dinamicamente con ros2 param set."""
        for param in params:
            self.get_logger().info(f'Actualizando parametro dinámico de camara "{param.name}" -> {param.value}')
            if param.name == 'min_contour_area':
                self.min_contour_area = float(param.value)
            elif param.name == 'warning_area':
                self.warning_area = float(param.value)
            elif param.name == 'danger_area':
                self.danger_area = float(param.value)
            elif param.name == 'roi_top_ratio':
                self.roi_top_ratio = float(param.value)
            elif param.name == 'publish_annotated_image':
                self.publish_annotated_image = bool(param.value)

        return SetParametersResult(successful=True)

    def _image_callback(self, msg: Image):
        """Procesa cada fotograma recibido de la camara."""
        try:
            # Conversion de mensaje ROS a imagen OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = cv_img.shape

            # Definicion de ROI: zona frontal de trayectoria (parte inferior de la imagen)
            roi_y_start = int(height * self.roi_top_ratio)
            roi = cv_img[roi_y_start:height, 0:width]

            # Preprocesamiento: escala de grises, blur y deteccion de bordes con Canny
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Operacion morfologica para agrupar contornos
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Busqueda de contornos
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0.0
            best_bbox = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.min_contour_area and area > max_area:
                    max_area = area
                    best_bbox = cv2.boundingRect(cnt)

            out_msg = ObstacleInfo()
            out_msg.header = msg.header
            out_msg.sensor_type = 'camera'

            if best_bbox is not None:
                bx, by, bw, bh = best_bbox
                # Reajustar coordenadas globales de la imagen completa
                global_by = by + roi_y_start

                out_msg.obstacle_detected = True
                # Estimacion aproximada de distancia inversamente proporcional al tamano visual
                estimated_distance = max(0.5, float(500.0 / math.sqrt(max_area)))
                out_msg.distance = estimated_distance

                # Normalizar posicion relativa en el campo de vision
                out_msg.x = estimated_distance
                out_msg.y = float((bx + bw / 2.0 - width / 2.0) / (width / 2.0))
                out_msg.z = 0.0

                if max_area >= self.danger_area:
                    out_msg.severity = 'DANGER'
                    box_color = (0, 0, 255)  # Rojo
                    self.get_logger().error(
                        f'[CAMARA-PELIGRO] Obstaculo visual grande (Area: {max_area:.0f}px, Dist est: {estimated_distance:.2f}m)',
                        throttle_duration_sec=1.0
                    )
                elif max_area >= self.warning_area:
                    out_msg.severity = 'WARNING'
                    box_color = (0, 165, 255)  # Naranja
                    self.get_logger().warn(
                        f'[CAMARA-AVISO] Obstaculo visual detectado (Area: {max_area:.0f}px, Dist est: {estimated_distance:.2f}m)',
                        throttle_duration_sec=1.0
                    )
                else:
                    out_msg.severity = 'SAFE'
                    box_color = (0, 255, 0)  # Verde

                if self.publish_annotated_image:
                    cv2.rectangle(cv_img, (bx, global_by), (bx + bw, global_by + bh), box_color, 3)
                    cv2.putText(
                        cv_img,
                        f'{out_msg.severity} - {estimated_distance:.1f}m ({int(max_area)}px)',
                        (bx, max(30, global_by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        box_color,
                        2
                    )
            else:
                out_msg.obstacle_detected = False
                out_msg.distance = float('inf')
                out_msg.x = 0.0
                out_msg.y = 0.0
                out_msg.z = 0.0
                out_msg.severity = 'SAFE'

            self.obstacle_pub.publish(out_msg)

            # Publicar imagen anotada para visualizacion si esta habilitado
            if self.publish_annotated_image:
                # Dibujar linea de inicio de la ROI
                cv2.line(cv_img, (0, roi_y_start), (width, roi_y_start), (255, 255, 0), 2)
                cv2.putText(
                    cv_img,
                    f'ROI Detector Camara | Estado: {out_msg.severity}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )
                annotated_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                annotated_msg.header = msg.header
                self.image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error procesando imagen: {str(e)}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = CameraObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
