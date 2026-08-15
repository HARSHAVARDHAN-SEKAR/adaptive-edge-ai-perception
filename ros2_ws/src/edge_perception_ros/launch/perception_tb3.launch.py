#!/usr/bin/env python3
"""Simulate: TurtleBot3 (waffle, has a camera) in Gazebo + the perception node.

Prereqs (ROS2 Humble example):
    sudo apt install ros-humble-turtlebot3-gazebo ros-humble-vision-msgs \
                     ros-humble-cv-bridge
    export TURTLEBOT3_MODEL=waffle

Run:
    ros2 launch edge_perception_ros perception_tb3.launch.py
    # drive the robot around:
    ros2 run turtlebot3_teleop teleop_keyboard
    # watch:
    ros2 topic echo /perception/status
    rviz2 -d $(ros2 pkg prefix edge_perception_ros)/share/edge_perception_ros/perception.rviz

If your simulated camera publishes elsewhere, override:
    ros2 launch edge_perception_ros perception_tb3.launch.py \
        camera_topic:=/my_robot/camera/image_raw use_gazebo:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    use_gazebo = LaunchConfiguration("use_gazebo")
    backend = LaunchConfiguration("backend")
    device = LaunchConfiguration("device")

    args = [
        DeclareLaunchArgument("camera_topic", default_value="/camera/image_raw",
                              description="simulated camera image topic"),
        DeclareLaunchArgument("use_gazebo", default_value="true",
                              description="also launch TurtleBot3 Gazebo world"),
        DeclareLaunchArgument("backend", default_value="real",
                              description="real | mock | auto"),
        DeclareLaunchArgument("device", default_value="cpu",
                              description="cpu | cuda"),
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="true under Gazebo (sim clock)"),
    ]

    actions = list(args)

    # -- TurtleBot3 world (optional; skip with use_gazebo:=false) ----------
    try:
        tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(tb3_gazebo, "launch",
                                 "turtlebot3_world.launch.py")),
                condition=IfCondition(use_gazebo),
            )
        )
    except Exception:
        # turtlebot3_gazebo not installed — perception node still launches;
        # point camera_topic at whatever sim/camera you have.
        pass

    # -- perception node ------------------------------------------------------
    actions.append(
        Node(
            package="edge_perception_ros",
            executable="perception_node",
            name="edge_perception",
            output="screen",
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time"),
                         "backend": backend, "device": device,
                         "adaptive": True}],
            remappings=[("/camera/image_raw", camera_topic)],
        )
    )
    return LaunchDescription(actions)
