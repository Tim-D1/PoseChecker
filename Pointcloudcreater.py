import os
import time
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
import sensor_msgs_py.point_cloud2 as pc2
import json

class PosePointCloudHandler(Node):
    def __init__(self):
        super().__init__('pose_pointcloud_handler')

        # Subscriber für Pose
        self.pose_sub = self.create_subscription(
            Pose, '/scout_pose', self.pose_callback, 10
        )
        # Publisher für Ready-Signal
        self.ready_pub = self.create_publisher(Empty, '/ready_for_next', 10)

        # Parameter
        self.msgs_needed = 10
        self.position_index = 1
        self.scan_index = 700
        self.positions_per_scan = 15

        # Basisordner
        self.base_output_dir = "Pointcloud"

        self.get_logger().info("Warte auf erste Pose-Nachricht ...")

        # State
        self.pose_received = False
        self.front_points = []
        self.rear_points = []
        self.current_pose = None

    def pose_callback(self, msg):
        if self.pose_received:
            return

        self.pose_received = True
        self.get_logger().info(f"Pose {self.position_index}/15 empfangen. Warte 1 Sekunden …")
        time.sleep(3.0)

        # Pose speichern
        self.current_pose = msg

        # leere Buffers
        self.front_points = []
        self.rear_points = []

        # Subscriber für Punktwolken starten
        self.front_sub = self.create_subscription(
            PointCloud2, '/point_cloud_front', self.front_callback, 10
        )
        self.rear_sub = self.create_subscription(
            PointCloud2, '/point_cloud_rear', self.rear_callback, 10
        )

    def front_callback(self, msg):
        if len(self.front_points) < self.msgs_needed:
            pts = self.extract_points(msg)
            if pts.size:
                self.front_points.append(pts)
                self.get_logger().info(f"Front PW #{len(self.front_points)} empfangen.")
            self.check_and_finalize()

    def rear_callback(self, msg):
        if len(self.rear_points) < self.msgs_needed:
            pts = self.extract_points(msg)
            if pts.size:
                self.rear_points.append(pts)
                self.get_logger().info(f"Rear PW #{len(self.rear_points)} empfangen.")
            self.check_and_finalize()

    def extract_points(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        return np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)

    def check_and_finalize(self):
        if len(self.front_points) < self.msgs_needed or len(self.rear_points) < self.msgs_needed:
            return

        self.get_logger().info("Genügend PW empfangen. Speichere …")
        front_all = np.vstack(self.front_points)
        rear_all  = np.vstack(self.rear_points)

        # Ordner: Pointcloud/posX/ScanY/
        pos_dir  = os.path.join(self.base_output_dir, f"pos{self.position_index}")
        scan_dir = os.path.join(pos_dir, f"Scan{self.scan_index}")
        os.makedirs(scan_dir, exist_ok=True)

        # Dateinamen
        front_fname = f"frontpos{self.position_index}scan{self.scan_index}.ply"
        rear_fname  = f"rearpos{self.position_index}scan{self.scan_index}.ply"

        # Speichern
        self._write_ply(front_all, os.path.join(scan_dir, front_fname))
        self._write_ply(rear_all,  os.path.join(scan_dir, rear_fname))
        self._write_pose(scan_dir)

        # Subs löschen
        self.destroy_subscription(self.front_sub)
        self.destroy_subscription(self.rear_sub)

        # Ready senden
        self.ready_pub.publish(Empty())
        self.get_logger().info("READY gesendet.")

        # Index-Logik
        self.position_index += 1
        if self.position_index > self.positions_per_scan:
            self.position_index = 1
            self.scan_index += 1
            self.get_logger().info(f"--- Neuer Scan-Index: {self.scan_index} gestartet ---")

        self.pose_received = False

    def _write_ply(self, points: np.ndarray, path: str):
        if points.size == 0:
            self.get_logger().warn(f"Keine Punkte, überspringe {path}")
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path, pcd)
        self.get_logger().info(f"Punktwolke gespeichert: {path}")

    def _write_pose(self, scan_dir: str):
        if self.current_pose is None:
            return
        pose_data = {
            'position': {
                'x': self.current_pose.position.x,
                'y': self.current_pose.position.y,
                'z': self.current_pose.position.z
            },
            'orientation': {
                'x': self.current_pose.orientation.x,
                'y': self.current_pose.orientation.y,
                'z': self.current_pose.orientation.z,
                'w': self.current_pose.orientation.w
            }
        }
        pose_path = os.path.join(scan_dir, "pose.json")
        with open(pose_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        self.get_logger().info(f"Pose gespeichert: {pose_path}")

def main(args=None):
    rclpy.init(args=args)
    node = PosePointCloudHandler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
