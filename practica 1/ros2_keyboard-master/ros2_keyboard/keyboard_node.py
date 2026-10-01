from ros2_keyboard.keyboard import Keyboard
import rclpy
from rclpy.parameter import Parameter


def main(args=None):
    print('Hi from ros2_keyboard.')
    rclpy.init(args=args)
    try:
        keyboard = Keyboard()
        rclpy.spin(keyboard)
    except KeyboardInterrupt:
        keyboard.shutdown('Manually interrupted')
    finally:
        keyboard.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
