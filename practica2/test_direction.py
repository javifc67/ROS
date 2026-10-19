import rosbag2_py, numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

frame = 0
while reader.has_next() and frame < 150:
    topic, data, t = reader.read_next()
    if topic == '/ouster/points':
        frame += 1
        if frame in [1, 20, 50, 100, 140]:
            msg = deserialize_message(data, PointCloud2)
            arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True)
            pts = np.column_stack((arr['x'], arr['y'], arr['z']))
            sub = pts[(np.abs(pts[:, 1]) < 2.0) & (pts[:, 2] > -0.5) & (pts[:, 2] < 1.5)]
            pos_x = sub[sub[:, 0] > 0.5]
            neg_x = sub[sub[:, 0] < -0.5]
            min_pos = np.min(pos_x[:, 0]) if len(pos_x) else -1
            min_neg = np.min(np.abs(neg_x[:, 0])) if len(neg_x) else -1
            print(f'Frame {frame}: pos_x count={len(pos_x)}, min_pos_x={min_pos:.2f} | neg_x count={len(neg_x)}, min_dist_neg={min_neg:.2f}')

