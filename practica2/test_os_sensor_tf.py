import rosbag2_py, numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

lidar_idx = 0
targets = {10, 450, 900}

while reader.has_next() and lidar_idx <= max(targets):
    topic, data, t = reader.read_next()
    if topic == '/ouster/points':
        lidar_idx += 1
        if lidar_idx in targets:
            msg = deserialize_message(data, PointCloud2)
            arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True)
            raw = np.column_stack((arr['x'], arr['y'], arr['z']))
            # In os_sensor:
            # x_sensor = -x_lidar
            # y_sensor = -y_lidar
            # z_sensor = z_lidar + 0.038195
            pts_sensor = np.column_stack((-raw[:, 0], -raw[:, 1], raw[:, 2] + 0.038195))
            
            # Forward sector in os_sensor: x > 1.0, |y| < 2.0, z in [-0.5, 1.5]
            fwd = pts_sensor[(pts_sensor[:, 0] > 1.0) & (pts_sensor[:, 0] < 30.0) & (np.abs(pts_sensor[:, 1]) < 2.0) & (pts_sensor[:, 2] > -0.5) & (pts_sensor[:, 2] < 1.5)]
            bwd = pts_sensor[(pts_sensor[:, 0] < -1.0) & (pts_sensor[:, 0] > -30.0) & (np.abs(pts_sensor[:, 1]) < 2.0) & (pts_sensor[:, 2] > -0.5) & (pts_sensor[:, 2] < 1.5)]
            print(f'LiDAR Frame {lidar_idx}:')
            print(f'  Forward (x_sensor > 1.0): count={len(fwd)}, min_x={np.min(fwd[:,0]) if len(fwd) else -1:.2f}m')
            print(f'  Backward (x_sensor < -1.0): count={len(bwd)}, min_dist={np.min(np.abs(bwd[:,0])) if len(bwd) else -1:.2f}m')

