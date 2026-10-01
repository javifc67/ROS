#!/usr/bin/env python
# coding=utf-8

from std_msgs.msg import Float64, Bool, String, Int64


class Topic(object):
    """
    Clase para almacenar los parametros que se vayan a necesitar para los topics de ROS
    name = Nombre del topic
    type = Tipo de datos del topic
    """

    __slots__ = ('name', 'type')

    def __init__(self, n, t):
        """
        Constructor para la clase topic
        @param n: Nombre del topic
        @type n: str
        @param t: Tipo de datos (std_msgs.msg)
        """

        self.name = n  # type: str
        self.type = t


Steering_Enabled = Topic('/can/steering/enable', Bool)
Throttle_Enabled = Topic('/can/throttle/enable', Bool)
Brake_Enabled = Topic('/can/brake/enable', Bool)
Waypoints_Start = Topic('/waypoints/start', Bool)
lidar_use = Topic('/decision/lidar/use', Bool)
Recorder_start = Topic('/recorder/start', Bool)
dump_box = Topic('/can/dump_box/command', Int64)

# Targets are not working because there is not a decision node
#Target_Speed = Topic('/can/speed/command', Float64)
#Target_Steering = Topic('/can/steering/command', Float64)