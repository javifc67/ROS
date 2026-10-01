#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.msg import String


class SubNode(Node):
    def __init__(self, name):
        super().__init__(name, namespace='introduccion')
        self.i = 0
        self.subs_msg = self.create_subscription(String, 'topic_name', self.callback, 10)

    def callback(self, msg: String):
        self.get_logger().info(msg.data)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = SubNode('sub')
        rclpy.spin(node)
        # nuestro código
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
