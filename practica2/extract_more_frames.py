import rosbag2_py, cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

bridge = CvBridge()
count = 0
targets = {400, 700, 1000, 1300}
while reader.has_next() and count <= max(targets):
    topic, data, t = reader.read_next()
    if topic == '/my_camera/pylon_ros2_camera_node/image_raw':
        count += 1
        if count in targets:
            msg = deserialize_message(data, Image)
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite(f'/home/sandy/ROS/practica2/cam_sample_{count}.jpg', img)
            print(f'Saved cam_sample_{count}.jpg')
