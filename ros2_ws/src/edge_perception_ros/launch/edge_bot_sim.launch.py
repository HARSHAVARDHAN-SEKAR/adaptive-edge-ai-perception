#!/usr/bin/env python3
"""Full simulation: spawn edge_bot (own URDF, camera-equipped) in Gazebo
and run the perception node against its camera.

Prereqs (Humble):
    sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-xacro \
        ros-humble-robot-state-publisher ros-humble-vision-msgs \
        ros-humble-cv-bridge

Run:
    ros2 launch edge_perception_ros edge_bot_sim.launch.py
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.2}, angular: {z: 0.3}}" -r 5     # drive
    ros2 topic echo /perception/status
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("edge_perception_ros")
    xacro_file = os.path.join(pkg, "urdf", "edge_bot.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]),
                                       value_type=str)
    use_sim_time = LaunchConfiguration("use_sim_time")
    backend = LaunchConfiguration("backend")

    actions = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("backend", default_value="auto",
                              description="real | mock | auto"),
    ]

    try:
        gazebo_ros = get_package_share_directory("gazebo_ros")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, "launch", "gazebo.launch.py"))))
    except Exception:
        pass  # allow perception-only bringup without Gazebo installed

    actions += [
        Node(package="robot_state_publisher",
             executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description,
                          "use_sim_time": use_sim_time}]),
        Node(package="gazebo_ros", executable="spawn_entity.py",
             arguments=["-topic", "robot_description",
                        "-entity", "edge_bot", "-z", "0.1"],
             output="screen"),
        Node(package="edge_perception_ros", executable="perception_node",
             name="edge_perception", output="screen",
             parameters=[{"use_sim_time": use_sim_time,
                          "backend": backend, "device": "cpu",
                          "adaptive": True}]),
    ]
    return LaunchDescription(actions)
