#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.msg import String


class TimerNode(Node):
    def __init__(self, name):
        super().__init__(name, namespace='introduccion')
        self.i = 0
        self.t = self.create_timer(0.1, self.t_callback)
        self.publisher = self.create_publisher(String, 'topic_name', 10)

    def t_callback(self):
        m = f'Hello world {self.i}'
        self.get_logger().warn(m)
        msg = String()
        msg.data = m
        self.publisher.publish(msg)
        self.i += 1


def main(args=None):
    try:
        rclpy.init(args=args)
        node = TimerNode('pub')
        rclpy.spin(node)
        # nuestro código
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
