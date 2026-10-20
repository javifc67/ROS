#!/usr/bin/env python3
"""
Nodo de deteccion de obstaculos mediante Camara (RGB / Imagen).
Implementa deteccion basada en Redes Neuronales Profundas (YOLOv8)
para clasificacion de vehiculos (coches, camiones, autobuses, motos)
y peatones, con estimacion de distancia y severidad de riesgo.
Incluye visualizacion grafica nativa 'Deteccion Camara' (OpenCV / Qt),
fallback automatico a contornos y publicacion de temas ROS 2.
"""

import math
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from obstacle_detector_interfaces.msg import ObstacleInfo

# Importacion condicional de Ultralytics YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class CameraObstacleDetector(Node):
    """
    Nodo que procesa imagenes RGB de la camara para deteccion visual de vehiculos y obstaculos.
    """

    # Clases de interes de COCO / KITTI (vehiculos y peatones)
    TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

    def __init__(self):
        super().__init__('camera_obstacle_detector')

        # Parametros configurables dinamicamente
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('model_path', '/home/jabibi/ROS/practica2/models/yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('warning_distance', 4.5)
        self.declare_parameter('danger_distance', 2.0)
        self.declare_parameter('min_contour_area', 3000.0)
        self.declare_parameter('warning_area', 25000.0)
        self.declare_parameter('danger_area', 70000.0)
        self.declare_parameter('roi_top_ratio', 0.35)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('show_gui_window', True)

        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameters_change)

        self.bridge = CvBridge()
        self.yolo_model = None

        # Inicializar modelo YOLO si esta habilitado y disponible
        if self.use_yolo and HAS_YOLO:
            self._init_yolo_model()

        # Suscriptor al topic de camara del bag
        self.subscription = self.create_subscription(
            Image,
            '/my_camera/pylon_ros2_camera_node/image_raw',
            self._image_callback,
            10
        )

        # Publicador de informacion de obstaculos
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

        modo = "YOLOv8 Deep Learning" if (self.use_yolo and self.yolo_model) else "OpenCV Contornos Clasico"
        self.get_logger().info(
            f'CameraObstacleDetector inicializado con exito. Modo: [{modo}]. '
            f'Umbral de Confianza: {self.confidence_threshold:.2f}, '
            f'Aviso: {self.warning_distance:.1f}m, Peligro: {self.danger_distance:.1f}m'
        )

    def _init_yolo_model(self):
        """Carga el modelo YOLO de deteccion de objetos."""
        try:
            model_target = self.model_path if os.path.exists(self.model_path) else 'yolov8n.pt'
            self.get_logger().info(f'Cargando modelo YOLO desde: {model_target}')
            self.yolo_model = YOLO(model_target)
            self.get_logger().info('Modelo YOLOv8 cargado correctamente.')
        except Exception as e:
            self.get_logger().warn(f'No se pudo cargar modelo YOLO ({str(e)}). Usando fallback de contornos.')
            self.yolo_model = None

    def _load_parameters(self):
        """Carga y almacena los parametros del nodo."""
        self.use_yolo = bool(self.get_parameter('use_yolo').value)
        self.model_path = str(self.get_parameter('model_path').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.warning_distance = float(self.get_parameter('warning_distance').value)
        self.danger_distance = float(self.get_parameter('danger_distance').value)
        self.min_contour_area = float(self.get_parameter('min_contour_area').value)
        self.warning_area = float(self.get_parameter('warning_area').value)
        self.danger_area = float(self.get_parameter('danger_area').value)
        self.roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self.publish_annotated_image = bool(self.get_parameter('publish_annotated_image').value)
        self.show_gui_window = bool(self.get_parameter('show_gui_window').value)

    def _on_parameters_change(self, params):
        """Callback ejecutado cuando se modifica un parametro dinamicamente con ros2 param set."""
        for param in params:
            self.get_logger().info(f'Actualizando parametro dinámico de camara "{param.name}" -> {param.value}')
            if param.name == 'use_yolo':
                self.use_yolo = bool(param.value)
                if self.use_yolo and self.yolo_model is None and HAS_YOLO:
                    self._init_yolo_model()
            elif param.name == 'confidence_threshold':
                self.confidence_threshold = float(param.value)
            elif param.name == 'warning_distance':
                self.warning_distance = float(param.value)
            elif param.name == 'danger_distance':
                self.danger_distance = float(param.value)
            elif param.name == 'min_contour_area':
                self.min_contour_area = float(param.value)
            elif param.name == 'warning_area':
                self.warning_area = float(param.value)
            elif param.name == 'danger_area':
                self.danger_area = float(param.value)
            elif param.name == 'roi_top_ratio':
                self.roi_top_ratio = float(param.value)
            elif param.name == 'publish_annotated_image':
                self.publish_annotated_image = bool(param.value)
            elif param.name == 'show_gui_window':
                self.show_gui_window = bool(param.value)

        return SetParametersResult(successful=True)

    def _detect_yolo(self, cv_img, height, width):
        """Deteccion de obstaculos mediante YOLOv8."""
        results = self.yolo_model(
            cv_img,
            classes=self.TARGET_CLASSES,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                area = bw * bh

                # Estimacion de distancia aproximada en funcion del tamano angular visual
                est_distance = max(0.5, float(500.0 / math.sqrt(area)))

                # Normalizar coordenada lateral respecto al centro del frame
                center_x = x1 + bw / 2.0
                rel_y = float((center_x - width / 2.0) / (width / 2.0))

                # Determinar severidad
                if est_distance <= self.danger_distance:
                    severity = 'DANGER'
                elif est_distance <= self.warning_distance:
                    severity = 'WARNING'
                else:
                    severity = 'SAFE'

                detections.append({
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(bw), int(bh)),
                    'distance': est_distance,
                    'x': est_distance,
                    'y': rel_y,
                    'severity': severity
                })

        return detections

    def _detect_contours(self, cv_img, height, width):
        """Fallback tradicional de deteccion por contornos morfológicos de OpenCV."""
        roi_y_start = int(height * self.roi_top_ratio)
        roi = cv_img[roi_y_start:height, 0:width]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                global_by = by + roi_y_start
                est_distance = max(0.5, float(500.0 / math.sqrt(area)))
                rel_y = float((bx + bw / 2.0 - width / 2.0) / (width / 2.0))

                if area >= self.danger_area or est_distance <= self.danger_distance:
                    severity = 'DANGER'
                elif area >= self.warning_area or est_distance <= self.warning_distance:
                    severity = 'WARNING'
                else:
                    severity = 'SAFE'

                detections.append({
                    'class_name': 'obstacle',
                    'confidence': 1.0,
                    'bbox': (bx, global_by, bw, bh),
                    'distance': est_distance,
                    'x': est_distance,
                    'y': rel_y,
                    'severity': severity
                })

        return detections

    def _image_callback(self, msg: Image):
        """Procesa cada fotograma recibido de la camara."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = cv_img.shape

            # Seleccionar algoritmo: YOLO o Contornos
            if self.use_yolo and self.yolo_model is not None:
                detections = self._detect_yolo(cv_img, height, width)
            else:
                detections = self._detect_contours(cv_img, height, width)

            out_msg = ObstacleInfo()
            out_msg.header = msg.header
            out_msg.sensor_type = 'camera_yolo' if (self.use_yolo and self.yolo_model) else 'camera_contour'

            if len(detections) > 0:
                # Ordenar por distancia (el mas cercano primero)
                detections.sort(key=lambda d: d['distance'])
                closest = detections[0]

                out_msg.obstacle_detected = True
                out_msg.distance = closest['distance']
                out_msg.x = closest['x']
                out_msg.y = closest['y']
                out_msg.z = 0.0

                # La severidad general es la mas critica entre las detecciones
                severities = [d['severity'] for d in detections]
                if 'DANGER' in severities:
                    out_msg.severity = 'DANGER'
                elif 'WARNING' in severities:
                    out_msg.severity = 'WARNING'
                else:
                    out_msg.severity = 'SAFE'

                # Logging informativo con throttling
                if out_msg.severity == 'DANGER':
                    self.get_logger().error(
                        f'[CAMARA-YOLO] Peligro crítico: {closest["class_name"]} a {closest["distance"]:.2f}m',
                        throttle_duration_sec=1.0
                    )
                elif out_msg.severity == 'WARNING':
                    self.get_logger().warn(
                        f'[CAMARA-YOLO] Aviso de obstaculo: {closest["class_name"]} a {closest["distance"]:.2f}m',
                        throttle_duration_sec=1.0
                    )

                # Anotaciones graficas en la imagen
                for det in detections:
                    bx, by, bw, bh = det['bbox']
                    sev = det['severity']

                    if sev == 'DANGER':
                        color = (0, 0, 255)  # Rojo
                    elif sev == 'WARNING':
                        color = (0, 165, 255)  # Naranja
                    else:
                        color = (255, 255, 0)  # Cyan / Amarillo

                    cv2.rectangle(cv_img, (bx, by), (bx + bw, by + bh), color, 2)

                    # Etiqueta con fondo sólido para máxima legibilidad
                    label = f"{det['class_name']} {det['confidence']:.2f} | {det['distance']:.1f}m"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    label_y = max(th + 5, by - 6)
                    cv2.rectangle(cv_img, (bx, label_y - th - 4), (bx + tw + 6, label_y + 4), color, -1)
                    cv2.putText(
                        cv_img,
                        label,
                        (bx + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
            else:
                out_msg.obstacle_detected = False
                out_msg.distance = float('inf')
                out_msg.x = 0.0
                out_msg.y = 0.0
                out_msg.z = 0.0
                out_msg.severity = 'SAFE'

            self.obstacle_pub.publish(out_msg)

            # Barra superior de estado (HUD)
            modo_str = "YOLOv8" if (self.use_yolo and self.yolo_model) else "Contornos"
            hud_color = (0, 0, 220) if out_msg.severity == 'DANGER' else ((0, 150, 255) if out_msg.severity == 'WARNING' else (0, 180, 0))
            cv2.rectangle(cv_img, (0, 0), (width, 42), (25, 25, 25), -1)
            cv2.line(cv_img, (0, 42), (width, 42), hud_color, 2)

            dist_text = f"{out_msg.distance:.2f}m" if out_msg.distance < 100 else "Libre"
            status_str = f"Modo: {modo_str} | Estado: {out_msg.severity} | Dist. mas cercana: {dist_text} | Obj: {len(detections)}"
            cv2.putText(cv_img, status_str, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Ventana interactiva nativa OpenCV 'Detección Cámara' (tal como en el video de la practica)
            if self.show_gui_window:
                cv2.imshow("Detección Cámara", cv_img)
                cv2.waitKey(1)

            # Publicar imagen anotada para RViz con frame_id 'os_lidar' para evitar errores de transformacion TF
            if self.publish_annotated_image:
                annotated_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                annotated_msg.header.stamp = msg.header.stamp
                annotated_msg.header.frame_id = 'os_lidar'
                self.image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error en detector de camara: {str(e)}', throttle_duration_sec=2.0)

    def destroy_node(self):
        if self.show_gui_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraObstacleDetector()
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
