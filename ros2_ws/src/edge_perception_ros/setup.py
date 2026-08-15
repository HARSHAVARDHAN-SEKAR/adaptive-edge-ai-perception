from setuptools import setup

package_name = "edge_perception_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "perception.rviz"]),
        ("share/" + package_name + "/launch",
         ["launch/perception_tb3.launch.py",
          "launch/edge_bot_sim.launch.py"]),
        ("share/" + package_name + "/urdf",
         ["urdf/edge_bot.urdf.xacro"]),
    ],
    install_requires=["setuptools", "edge-perception"],
    zip_safe=True,
    description="Adaptive Edge AI Perception Engine - ROS2 nodes",
    license="AGPL-3.0",
    entry_points={
        "console_scripts": [
            "perception_node = edge_perception_ros.perception_node:main",
        ],
    },
)
