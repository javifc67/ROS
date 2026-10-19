import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image
from tf2_msgs.msg import TFMessage

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

topics_to_find = {'/ouster/points', '/my_camera/pylon_ros2_camera_node/image_raw', '/ouster/nearir_image', '/tf_static'}
found = set()

while reader.has_next() and len(found) < len(topics_to_find):
    topic, data, t = reader.read_next()
    if topic in topics_to_find and topic not in found:
        found.add(topic)
        if topic == '/ouster/points':
            msg = deserialize_message(data, PointCloud2)
            print(f'=== {topic} ===')
            print(f'frame_id: {msg.header.frame_id}, width: {msg.width}, height: {msg.height}, point_step: {msg.point_step}')
            print('fields:', [f.name for f in msg.fields])
        elif topic in ('/my_camera/pylon_ros2_camera_node/image_raw', '/ouster/nearir_image'):
            msg = deserialize_message(data, Image)
            print(f'=== {topic} ===')
            print(f'frame_id: {msg.header.frame_id}, width: {msg.width}, height: {msg.height}, encoding: {msg.encoding}')
        elif topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            print(f'=== {topic} ===')
            for tr in msg.transforms:
                print(f'parent: {tr.header.frame_id} -> child: {tr.child_frame_id}')
                print(f'  trans: {tr.transform.translation.x}, {tr.transform.translation.y}, {tr.transform.translation.z}')
                print(f'  rot: {tr.transform.rotation.x}, {tr.transform.rotation.y}, {tr.transform.rotation.z}, {tr.transform.rotation.w}')

