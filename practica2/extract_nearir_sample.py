import rosbag2_py, cv2, numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

bridge = CvBridge()
count = 0
targets = {10, 200, 500}

while reader.has_next() and count <= max(targets):
    topic, data, t = reader.read_next()
    if topic == '/ouster/nearir_image':
        count += 1
        if count in targets:
            msg = deserialize_message(data, Image)
            # mono16 image
            raw_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            print(f'NearIR {count}: shape={raw_img.shape}, dtype={raw_img.dtype}, min={raw_img.min()}, max={raw_img.max()}, mean={raw_img.mean():.1f}')
            
            # Normalize to 8-bit with percentile clipping for good contrast
            p2, p98 = np.percentile(raw_img, (2, 98))
            norm = np.clip((raw_img.astype(np.float32) - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            # Apply colormap (e.g. cv2.COLORMAP_INFERNO or COLORMAP_TURBO or COLORMAP_VIRIDIS or keep monochrome/heat)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            
            cv2.imwrite(f'/home/sandy/ROS/practica2/nearir_{count}_gray.jpg', norm)
            cv2.imwrite(f'/home/sandy/ROS/practica2/nearir_{count}_inferno.jpg', colored)
            print(f'Saved nearir_{count}_gray.jpg and nearir_{count}_inferno.jpg')

