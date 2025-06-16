#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
import json
import os
import numpy as np

class JSONPosePublisher(Node):
    def __init__(self, poses, repeats=7):
        super().__init__("scout_pose_json_publisher")
        self.publisher_ = self.create_publisher(Pose, "scout_pose", 10)
        self.ready_sub = self.create_subscription(Empty, "/ready_for_next", self.ready_callback, 10)

        self.poses = poses
        self.repeats = repeats
        self.current_repeat = 0
        self.current_index = 0

        self.get_logger().info(f"JSONPosePublisher gestartet mit {len(self.poses)} Posen, Wiederholungen: {self.repeats}")

        # Erste Pose direkt senden
        self.publish_current_pose()

    def ready_callback(self, msg):
        self.current_index += 1

        if self.current_index >= len(self.poses):
            self.current_repeat += 1
            if self.current_repeat >= self.repeats:
                self.get_logger().info("Alle Wiederholungen abgeschlossen. Beende Node.")
                rclpy.shutdown()
                return
            else:
                self.get_logger().info(f"--- Wiederholung {self.current_repeat + 1}/{self.repeats} ---")
                self.current_index = 0

        self.publish_current_pose()

    def publish_current_pose(self):
        entry = self.poses[self.current_index]
        x = entry.get("x", 0.0)
        y = entry.get("y", 0.0)
        rot_x_deg = entry.get("rot_x_deg", 0.0)

        # === Verrauschung ===
        
        xnoise = np.random.normal(0.0, 0.05)
        x += xnoise
        ynoise = np.random.normal(0.0, 0.05)
        y += ynoise
        
        if rot_x_deg < 0:
            rot_x_deg += np.abs(np.random.normal(loc=0.0, scale=4.0))
        else:
            rot_x_deg -= np.abs(np.random.normal(loc=0.0, scale=4.0))
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.1

        rot = R.from_euler('x', rot_x_deg, degrees=True)
        q = rot.as_quat()

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        self.publisher_.publish(pose)
        self.get_logger().info(
            f"[{self.current_repeat + 1}/{self.repeats}] Pose {self.current_index + 1}/{len(self.poses)} gesendet: "
            f"x={x:.3f}, y={y:.3f}, rot_x={rot_x_deg:.2f}°"
        )

def main():
    json_file = "position2.json"
    if not os.path.isfile(json_file):
        print(f"Datei {json_file} nicht gefunden.")
        return

    with open(json_file, "r") as f:
        poses = json.load(f)

    rclpy.init()
    node = JSONPosePublisher(poses, repeats=1)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
