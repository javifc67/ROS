#!/usr/bin/env python3
"""
Nodo de control avanzado de Turtlesim para la Práctica 1 de Robótica.

Características:
- Movimiento continuo con W, A, S, D (soporta giro y avance simultáneo).
- SPACE: Activar / desactivar lápiz (/turtle1/set_pen).
- C: Borrar trazos de la pantalla (/clear).
- R: Reiniciar tortuga al centro y orientada hacia arriba (/turtle1/teleport_absolute).
- Parámetros dinámicos: 'speed', 'angular_speed' y 'log_level' modificables en tiempo de ejecución.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from turtlesim.srv import SetPen, TeleportAbsolute

from pynput import keyboard


class TeleopTurtle(Node):
    def __init__(self):
        super().__init__('teleop_turtle')

        # -------------------------------------------------------------
        # 1. Declaración y configuración de parámetros
        # -------------------------------------------------------------
        self.declare_parameter('speed', 2.0)
        self.declare_parameter('angular_speed', 2.0)
        self.declare_parameter('log_level', 'INFO')

        self.speed = float(self.get_parameter('speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.apply_log_level(self.get_parameter('log_level').value)

        # Callback para reaccionar a cambios de parámetros en tiempo de ejecución
        self.add_on_set_parameters_callback(self.on_parameter_change)

        # -------------------------------------------------------------
        # 2. Publicador de velocidad
        # -------------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

        # -------------------------------------------------------------
        # 3. Clientes de Servicio de Turtlesim
        # -------------------------------------------------------------
        self.set_pen_client = self.create_client(SetPen, '/turtle1/set_pen')
        self.clear_client = self.create_client(Empty, '/clear')
        self.teleport_client = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')

        # Estado interno
        self.pressed_keys = set()
        self.pen_enabled = True  # Por defecto el lápiz pinta en Turtlesim

        # -------------------------------------------------------------
        # 4. Timer para publicación continua de velocidad (30 Hz)
        # -------------------------------------------------------------
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # -------------------------------------------------------------
        # 5. Listener de teclado no bloqueante en hilo separado (pynput)
        # -------------------------------------------------------------
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.listener.daemon = True
        self.listener.start()

        self.print_instructions()

    def print_instructions(self):
        msg = (
            "\n"
            "====================================================\n"
            "   CONTROL DE TURTLESIM - PRÁCTICA 1\n"
            "====================================================\n"
            "  [W] / [S]        : Avanzar / Retroceder\n"
            "  [A] / [D]        : Girar Izquierda / Derecha\n"
            "  (Combina W+A o W+D para avanzar y girar a la vez)\n"
            "  [ESPACIO]        : Activar / Desactivar trazado\n"
            "  [C]              : Borrar pantalla (Clear)\n"
            "  [R]              : Reiniciar al centro y mirando arriba\n"
            "  [ESC] o Ctrl+C   : Salir\n"
            "====================================================\n"
            f" Velocidad lineal inicial  : {self.speed}\n"
            f" Velocidad angular inicial : {self.angular_speed}\n"
            "====================================================\n"
        )
        self.get_logger().info(msg)

    # -----------------------------------------------------------------
    # Gestión de Parámetros
    # -----------------------------------------------------------------
    def apply_log_level(self, level_str: str):
        level_str = str(level_str).upper()
        levels = {
            'DEBUG': rclpy.logging.LoggingSeverity.DEBUG,
            'INFO': rclpy.logging.LoggingSeverity.INFO,
            'WARN': rclpy.logging.LoggingSeverity.WARN,
            'ERROR': rclpy.logging.LoggingSeverity.ERROR,
            'FATAL': rclpy.logging.LoggingSeverity.FATAL,
        }
        severity = levels.get(level_str, rclpy.logging.LoggingSeverity.INFO)
        self.get_logger().set_level(severity)
        self.get_logger().info(f"Nivel de logging establecido a: {level_str}")

    def on_parameter_change(self, params):
        for param in params:
            if param.name == 'speed':
                self.speed = float(param.value)
                self.get_logger().info(f"Velocidad lineal actualizada a: {self.speed}")
            elif param.name == 'angular_speed':
                self.angular_speed = float(param.value)
                self.get_logger().info(f"Velocidad angular actualizada a: {self.angular_speed}")
            elif param.name == 'log_level':
                self.apply_log_level(param.value)
        return SetParametersResult(successful=True)

    # -----------------------------------------------------------------
    # Eventos de teclado (pynput)
    # -----------------------------------------------------------------
    def _extract_char(self, key):
        try:
            return key.char.lower() if key.char is not None else None
        except AttributeError:
            return None

    def on_key_press(self, key):
        char = self._extract_char(key)

        # Teclas de movimiento continuas (W, A, S, D)
        if char in ('w', 'a', 's', 'd'):
            self.pressed_keys.add(char)

        # Barra espaciadora: Habilitar / Deshabilitar trazo
        elif key == keyboard.Key.space:
            self.toggle_pen()

        # Tecla C: Borrar pantalla
        elif char == 'c':
            self.call_clear()

        # Tecla R: Reiniciar al centro mirando arriba
        elif char == 'r':
            self.call_reset_center_up()

        # Tecla ESC: Salida
        elif key == keyboard.Key.esc:
            self.get_logger().info("Saliendo del nodo de control...")
            rclpy.shutdown()

    def on_key_release(self, key):
        char = self._extract_char(key)
        if char in self.pressed_keys:
            self.pressed_keys.discard(char)

    # -----------------------------------------------------------------
    # Bucle periódico de publicación de movimiento (Permite giro+avance)
    # -----------------------------------------------------------------
    def timer_callback(self):
        twist = Twist()
        linear = 0.0
        angular = 0.0

        if 'w' in self.pressed_keys:
            linear += self.speed
        if 's' in self.pressed_keys:
            linear -= self.speed
        if 'a' in self.pressed_keys:
            angular += self.angular_speed
        if 'd' in self.pressed_keys:
            angular -= self.angular_speed

        twist.linear.x = float(linear)
        twist.angular.z = float(angular)

        # Publicar continuamente el estado de velocidad actual
        self.cmd_vel_pub.publish(twist)

        if linear != 0.0 or angular != 0.0:
            self.get_logger().debug(f"cmd_vel -> linear.x={twist.linear.x:.2f}, angular.z={twist.angular.z:.2f}")

    # -----------------------------------------------------------------
    # Servicios: SetPen, Clear, Reset / Teleport
    # -----------------------------------------------------------------
    def toggle_pen(self):
        if not self.set_pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Servicio /turtle1/set_pen no disponible.")
            return

        self.pen_enabled = not self.pen_enabled
        req = SetPen.Request()
        # off: 0 activa el trazo, 1 lo desactiva
        req.off = 0 if self.pen_enabled else 1
        req.r = 255
        req.g = 255
        req.b = 255
        req.width = 3

        estado = "HABILITADO" if self.pen_enabled else "DESHABILITADO"
        self.get_logger().info(f"[ESPACIO] Trazado de pintura {estado}")
        self.set_pen_client.call_async(req)

    def call_clear(self):
        if not self.clear_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Servicio /clear no disponible.")
            return

        self.get_logger().info("[C] Borrando trazado de la pantalla...")
        req = Empty.Request()
        self.clear_client.call_async(req)

    def call_reset_center_up(self):
        if not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Servicio /turtle1/teleport_absolute no disponible.")
            return

        self.get_logger().info("[R] Reiniciando tortuga al centro (5.54, 5.54) y orientada hacia ARRIBA...")

        req = TeleportAbsolute.Request()
        req.x = 5.5444445
        req.y = 5.5444445
        # pi / 2 radianes = 90 grados (orientación apuntando verticalmente hacia arriba)
        req.theta = math.pi / 2.0

        self.teleport_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopTurtle()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.listener.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
