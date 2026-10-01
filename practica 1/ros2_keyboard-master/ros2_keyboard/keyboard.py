#!/usr/bin/env python
# coding=utf-8

from rclpy.qos import HistoryPolicy
from rclpy.node import Node
from std_msgs.msg import Bool, Int64
from rclpy.parameter import Parameter
from ros2_keyboard.keyboard_topics import *
from pynput import keyboard

box_position = {
            1: 'bajar rápida',
            2: 'bajar media',
            3: 'bajar lenta',
            4: 'dinamico',
            5: 'estatico',
            6: 'subir lenta',
            7: 'subir media',
            8: 'subir rápida',
        }

class Keyboard(Node):
    """
    Class Keyboard.
    Main script to start Keyboard node
    """

    def __init__(self):

        super().__init__(node_name='keyboard', namespace='keyboard_name', allow_undeclared_parameters=True,
                         start_parameter_services=True, automatically_declare_parameters_from_overrides=True)

        self.throttle_active = False
        self.brake_active = False
        self.steering_active = False
        self.waypoints_active = False
        self.use_lidar = False
        self.rc_start = False
        self.logger = self.get_logger()
        self._log_level = self.get_parameter_or('log_level', Parameter(name='log_level', value=10)).value
        self.logger.set_level(self._log_level)
        self.update_timer = None
        self.timer = None
        self.dump_box_command = 4
        # --------------------------------------------------------------------------------------------------------------
        # Create publishers
        # --------------------------------------------------------------------------------------------------------------
        self.enable_steering_publisher = self.create_publisher(msg_type=Steering_Enabled.type,
                                                               topic=Steering_Enabled.name,
                                                               qos_profile=HistoryPolicy.KEEP_LAST)
        self.enable_throttle_publisher = self.create_publisher(msg_type=Throttle_Enabled.type,
                                                               topic=Throttle_Enabled.name,
                                                               qos_profile=HistoryPolicy.KEEP_LAST)
        self.enable_brake_publisher = self.create_publisher(msg_type=Brake_Enabled.type, topic=Brake_Enabled.name,
                                                            qos_profile=HistoryPolicy.KEEP_LAST)
        self.enable_waypoints_publisher = self.create_publisher(msg_type=Waypoints_Start.type,
                                                                topic=Waypoints_Start.name,
                                                                qos_profile=HistoryPolicy.KEEP_LAST)
        self.use_lidar_publisher = self.create_publisher(msg_type=lidar_use.type, topic=lidar_use.name,
                                                         qos_profile=HistoryPolicy.KEEP_LAST)
        self.recorder_start_publisher = self.create_publisher(msg_type=Recorder_start.type, topic=Recorder_start.name,
                                                              qos_profile=HistoryPolicy.KEEP_LAST)
        self.dump_box_publisher = self.create_publisher(msg_type=dump_box.type, topic=dump_box.name,
                                                        qos_profile=HistoryPolicy.KEEP_LAST)

        self.logger.info('Start keyboard')
        timer_period = self.get_parameter_or('freq', Parameter(name='freq', value=10)).value
        self.update_timer = self.create_timer(1 / timer_period, self.update)
        self.keyboard_start()

    def shutdown(self, reason=None):
        self.logger.info('{} Node is DOWN'.format(self.get_name()))
        if reason:
            self.logger.info(reason)

    def keyboard_start(self):
        lis = keyboard.Listener(on_press=self.on_press)
        lis.start()  # start to listen on a separate thread
        self.logger.info("\033[2J\033[1;1f")
        self.logger.info('Legend keys')
        # self.logger.info('Arrows/asdw Control speed and steering')
        # self.logger.info('p/space-> stop steering 0 speed 0')
        # self.logger.info('U->Throttle I->Brake O->Steering K/L-> ALL OFF/ON W-> Waypoints')
        self.logger.info('1->Steering 2->Brake 3->Throttle')
        self.logger.info('5->Waypoints 6->-No P Lidar- 7->Recorder')
        self.logger.info('8->box down 9->box up')
        self.logger.info('\033[s')
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.print_info)

    def print_info(self):
        # self.logger.info("\033[5;1f")
        self.logger.info("\033[u")
        self.logger.info(
            'Throttle {}  Brake {}  Steering {}\033[K'.format(self.throttle_active, self.brake_active,
                                                              self.steering_active))
        self.logger.info(
            'Waypoints {} Lidar {} record {}\033[K'.format(self.waypoints_active, self.use_lidar, self.rc_start))
        self.logger.info(
            'Dump_box_command {}\033[K'.format(box_position.get(self.dump_box_command)))

    def on_press(self, key):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys
        if key == keyboard.Key.esc:
            return False  # stop listener
        elif key == keyboard.Key.space:
            self.throttle_active = False
            self.brake_active = False
            self.steering_active = False
        # elif k == 'l':
        #     self.throttle_active = True
        #     self.brake_active = True
        #     self.steering_active = True
        elif k == keyboard.Key.f3 or k == '3':
            if self.throttle_active:
                self.throttle_active = False
            else:
                self.throttle_active = True
        elif k == keyboard.Key.f2 or k == '2':
            if self.brake_active:
                self.brake_active = False
            else:
                self.brake_active = True
        elif k == keyboard.Key.f1 or k == '1':
            if self.steering_active:
                self.steering_active = False
            else:
                self.steering_active = True
        elif k == keyboard.Key.f5 or k == '5':
            if self.waypoints_active:
                self.waypoints_active = False
            else:
                self.waypoints_active = True
        elif k == keyboard.Key.f6 or k == '6':
            if self.use_lidar:
                self.use_lidar = False
            else:
                self.use_lidar = True
        elif k == keyboard.Key.f7 or k == '7':
            if self.rc_start:
                self.rc_start = False
            else:
                self.rc_start = True
        elif k == keyboard.Key.f8 or k == '8':
            if self.dump_box_command > 1:
                self.dump_box_command -= 1
        elif k == keyboard.Key.f9 or k == '9':
            if self.dump_box_command < 8:
                self.dump_box_command += 1

    def update(self):
        self.enable_throttle_publisher.publish(Bool(data=self.throttle_active))
        self.enable_brake_publisher.publish(Bool(data=self.brake_active))
        self.enable_steering_publisher.publish(Bool(data=self.steering_active))
        self.enable_waypoints_publisher.publish(Bool(data=self.waypoints_active))
        # self.use_lidar_publisher.publish(Bool(data=self.use_lidar))
        self.recorder_start_publisher.publish(Bool(data=self.rc_start))
        self.dump_box_publisher.publish(Int64(data=self.dump_box_command))

    # def on_press(self, key):
    #     try:
    #         k = key.char  # single-char keys
    #     except:
    #         k = key.name  # other keys
    #     if key == keyboard.Key.esc:
    #         return False  # stop listener
    #     elif k == 'k' or key == keyboard.Key.space:
    #         self.throttle_active = False
    #         self.brake_active = False
    #         self.steering_active = False
    #     elif k == 'l':
    #         self.throttle_active = True
    #         self.brake_active = True
    #         self.steering_active = True
    #     elif k == 'u':
    #         if self.throttle_active:
    #             self.throttle_active = False
    #         else:
    #             self.throttle_active = True
    #     elif k == 'i':
    #         if self.brake_active:
    #             self.brake_active = False
    #         else:
    #             self.brake_active = True
    #     elif k == 'o':
    #         if self.steering_active:
    #             self.steering_active = False
    #         else:
    #             self.steering_active = True
    #     elif k == 'w':
    #         if self.waypoints_active:
    #             self.waypoints_active = False
    #         else:
    #             self.waypoints_active = True
    #     elif k == 'e':
    #         if self.use_lidar:
    #             self.use_lidar = False
    #         else:
    #             self.use_lidar = True
