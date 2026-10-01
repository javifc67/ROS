#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


def main(args=None):
    try:
        rclpy.init(args=args)
        node = Node('logger')
        node.get_logger().warn('Hello world')
        rclpy.spin(node)
        # nuestro código
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
