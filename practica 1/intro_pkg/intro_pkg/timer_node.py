#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


class TimerNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.i = 0
        self.t = self.create_timer(0.5, self.t_callback)

    def t_callback(self):
        self.get_logger().warn(f'Hello world {self.i}')
        self.i += 1


def main(args=None):
    try:
        rclpy.init(args=args)
        node = TimerNode('timer')
        rclpy.spin(node)
        # nuestro código
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
