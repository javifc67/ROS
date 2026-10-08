#!/usr/bin/env python3
"""
Nodo gestor de seguridad y fusion sensorial de obstaculos.
Recibe detecciones de LIDAR y camara, fusiona la informacion para determinar
el nivel de riesgo global del vehiculo, publica el estado en /safety/status
y expone un SERVICIO ROS 2 (/safety/query_status) para consultas sincronas.
Soporta reconfiguracion de parametros en tiempo de ejecucion.
"""

import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from obstacle_detector_interfaces.msg import ObstacleInfo
from obstacle_detector_interfaces.srv import QuerySafetyStatus


class ObstacleManager(Node):
    """
    Nodo de fusion sensorial y gestion de la seguridad del vehiculo.
    """

    def __init__(self):
        super().__init__('obstacle_manager')

        # Parametros configurables dinamicamente
        self.declare_parameter('emergency_distance', 1.5)
        self.declare_parameter('warning_distance', 3.0)
        self.declare_parameter('fusion_mode', 'pessimistic')  # 'pessimistic' o 'lidar_priority'
        self.declare_parameter('sensor_timeout_sec', 1.5)

        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameters_change)

        # Ultimas lecturas de los sensores
        self.last_lidar_msg = None
        self.last_lidar_time = 0.0

        self.last_camera_msg = None
        self.last_camera_time = 0.0

        # Suscriptores a los dos sensores (LIDAR y Camara)
        self.sub_lidar = self.create_subscription(
            ObstacleInfo,
            '/obstacles/lidar',
            self._lidar_callback,
            10
        )

        self.sub_camera = self.create_subscription(
            ObstacleInfo,
            '/obstacles/camera',
            self._camera_callback,
            10
        )

        # Publicador de estado de seguridad global unificado
        self.status_pub = self.create_publisher(
            ObstacleInfo,
            '/safety/status',
            10
        )

        # SERVICIO ROS 2 propio: Permite consultar el estado de seguridad
        self.srv = self.create_service(
            QuerySafetyStatus,
            '/safety/query_status',
            self._handle_query_status
        )

        # Timer periodico para evaluar y publicar el estado a 10 Hz
        self.timer = self.create_timer(0.1, self._fusion_timer_callback)

        self.get_logger().info(
            'ObstacleManager inicializado correctamente. '
            'Servicio /safety/query_status disponible. '
            f'Modo de fusion: {self.fusion_mode}, '
            f'Distancia de emergencia: {self.emergency_distance:.1f}m'
        )

    def _load_parameters(self):
        """Carga y almacena los parametros del nodo."""
        self.emergency_distance = float(self.get_parameter('emergency_distance').value)
        self.warning_distance = float(self.get_parameter('warning_distance').value)
        self.fusion_mode = str(self.get_parameter('fusion_mode').value)
        self.sensor_timeout_sec = float(self.get_parameter('sensor_timeout_sec').value)

    def _on_parameters_change(self, params):
        """Callback ejecutado cuando se modifica un parametro dinamicamente con ros2 param set."""
        for param in params:
            self.get_logger().info(f'Actualizando parametro dinámico de gestion "{param.name}" -> {param.value}')
            if param.name == 'emergency_distance':
                self.emergency_distance = float(param.value)
            elif param.name == 'warning_distance':
                self.warning_distance = float(param.value)
            elif param.name == 'fusion_mode':
                self.fusion_mode = str(param.value)
            elif param.name == 'sensor_timeout_sec':
                self.sensor_timeout_sec = float(param.value)

        return SetParametersResult(successful=True)

    def _lidar_callback(self, msg: ObstacleInfo):
        """Almacena la ultima medicion de LIDAR recibida."""
        self.last_lidar_msg = msg
        self.last_lidar_time = time.time()

    def _camera_callback(self, msg: ObstacleInfo):
        """Almacena la ultima medicion de Camara recibida."""
        self.last_camera_msg = msg
        self.last_camera_time = time.time()

    def _fusion_timer_callback(self):
        """Fusiona periodicamente las detecciones de ambos sensores y publica el estado."""
        now = time.time()
        lidar_active = (self.last_lidar_msg is not None) and ((now - self.last_lidar_time) < self.sensor_timeout_sec)
        camera_active = (self.last_camera_msg is not None) and ((now - self.last_camera_time) < self.sensor_timeout_sec)

        unified_msg = ObstacleInfo()
        unified_msg.header.stamp = self.get_clock().now().to_msg()
        unified_msg.header.frame_id = 'base_link'
        unified_msg.sensor_type = 'fusion_lidar_camera'

        if not lidar_active and not camera_active:
            unified_msg.obstacle_detected = False
            unified_msg.distance = float('inf')
            unified_msg.severity = 'SAFE'
            self.status_pub.publish(unified_msg)
            return

        min_distance = float('inf')
        severities = []
        best_x = 0.0
        best_y = 0.0
        best_z = 0.0

        if lidar_active and self.last_lidar_msg.obstacle_detected:
            severities.append(self.last_lidar_msg.severity)
            if self.last_lidar_msg.distance < min_distance:
                min_distance = self.last_lidar_msg.distance
                best_x = self.last_lidar_msg.x
                best_y = self.last_lidar_msg.y
                best_z = self.last_lidar_msg.z

        if camera_active and self.last_camera_msg.obstacle_detected:
            severities.append(self.last_camera_msg.severity)
            if self.last_camera_msg.distance < min_distance:
                min_distance = self.last_camera_msg.distance
                # Si el LIDAR no reporto coordenadas 3D, usamos la estimacion de la camara
                if best_x == 0.0 and best_y == 0.0:
                    best_x = self.last_camera_msg.x
                    best_y = self.last_camera_msg.y

        # Logica de determinacion de severidad unificada
        if 'DANGER' in severities or (min_distance <= self.emergency_distance):
            unified_severity = 'DANGER'
        elif 'WARNING' in severities or (min_distance <= self.warning_distance):
            unified_severity = 'WARNING'
        else:
            unified_severity = 'SAFE'

        unified_msg.obstacle_detected = (len(severities) > 0)
        unified_msg.distance = min_distance if unified_msg.obstacle_detected else float('inf')
        unified_msg.x = best_x
        unified_msg.y = best_y
        unified_msg.z = best_z
        unified_msg.severity = unified_severity

        self.status_pub.publish(unified_msg)

    def _handle_query_status(self, request, response):
        """
        Manejador del SERVICIO /safety/query_status.
        Responde con el diagnostico de seguridad segun el filtro solicitado.
        """
        sensor = request.sensor_filter.lower().strip()
        now = time.time()

        self.get_logger().info(f'Solicitud de servicio recibida: filtro="{sensor}"')

        if sensor == 'lidar':
            if self.last_lidar_msg and (now - self.last_lidar_time < self.sensor_timeout_sec):
                msg = self.last_lidar_msg
                response.closest_distance = msg.distance
                response.alert_level = msg.severity
                response.safe_to_proceed = (msg.severity != 'DANGER')
                response.message = f'LIDAR OK. Distancia: {msg.distance:.2f}m'
            else:
                response.closest_distance = float('inf')
                response.alert_level = 'NO_DATA'
                response.safe_to_proceed = False
                response.message = 'No se reciben datos recientes de LIDAR'

        elif sensor == 'camera':
            if self.last_camera_msg and (now - self.last_camera_time < self.sensor_timeout_sec):
                msg = self.last_camera_msg
                response.closest_distance = msg.distance
                response.alert_level = msg.severity
                response.safe_to_proceed = (msg.severity != 'DANGER')
                response.message = f'Camara OK. Distancia estimada: {msg.distance:.2f}m'
            else:
                response.closest_distance = float('inf')
                response.alert_level = 'NO_DATA'
                response.safe_to_proceed = False
                response.message = 'No se reciben datos recientes de Camara'

        else:  # 'all' o cualquier otro valor -> Evaluacion de fusion conjunta
            lidar_ok = self.last_lidar_msg and (now - self.last_lidar_time < self.sensor_timeout_sec)
            camera_ok = self.last_camera_msg and (now - self.last_camera_time < self.sensor_timeout_sec)

            min_dist = float('inf')
            is_danger = False
            is_warning = False

            if lidar_ok and self.last_lidar_msg.obstacle_detected:
                min_dist = min(min_dist, self.last_lidar_msg.distance)
                if self.last_lidar_msg.severity == 'DANGER':
                    is_danger = True
                elif self.last_lidar_msg.severity == 'WARNING':
                    is_warning = True

            if camera_ok and self.last_camera_msg.obstacle_detected:
                min_dist = min(min_dist, self.last_camera_msg.distance)
                if self.last_camera_msg.severity == 'DANGER':
                    is_danger = True
                elif self.last_camera_msg.severity == 'WARNING':
                    is_warning = True

            if is_danger or min_dist <= self.emergency_distance:
                response.alert_level = 'DANGER'
                response.safe_to_proceed = False
                response.message = f'ALERTA ROJA: Obstaculo critico a {min_dist:.2f}m. Detencion requerida.'
            elif is_warning or min_dist <= self.warning_distance:
                response.alert_level = 'WARNING'
                response.safe_to_proceed = True
                response.message = f'PRECAUCION: Obstaculo detectado a {min_dist:.2f}m. Reducir velocidad.'
            else:
                response.alert_level = 'SAFE'
                response.safe_to_proceed = True
                response.message = 'Camino despejado. Seguro para continuar navegando.'

            response.closest_distance = min_dist

        self.get_logger().info(
            f'Respuesta de servicio enviada: Seguro={response.safe_to_proceed}, '
            f'Nivel={response.alert_level}, Distancia={response.closest_distance:.2f}m'
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
