import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile


class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_fixer')
        self.pub = self.create_publisher(CameraInfo, '/camera_info_fixed', QoSProfile(depth=10))
        self.timer = self.create_timer(0.05, self.publish_info)  # 20Hz

    def publish_info(self):
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.width = 320
        msg.height = 240
        msg.distortion_model = 'plumb_bob'
        msg.k = [
            245.574224, 0.0,        164.427251,
            0.0,        246.335295,  99.800133,
            0.0,        0.0,          1.0
        ]
        msg.d = [-0.036978, 0.008770, -0.041104, 0.012788, 0.0]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        msg.p = [
            247.309906, 0.0,        168.408758, 0.0,
            0.0,        239.618699,  89.403075, 0.0,
            0.0,        0.0,          1.0,      0.0
        ]
        self.pub.publish(msg)


rclpy.init()
rclpy.spin(CameraInfoPublisher())
