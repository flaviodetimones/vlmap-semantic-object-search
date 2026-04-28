"""Offline contract checks for the HSR Gazebo bringup profile."""

from pathlib import Path
import importlib.util
from importlib.machinery import SourceFileLoader
import sys
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRINGUP = ROOT / "ros_ws" / "src" / "vlmap_bringup"
sys.path.insert(0, str(BRINGUP / "scripts"))


def test_hsr_launch_is_valid_xml_and_keeps_vlmap_stack():
    launch_path = BRINGUP / "launch" / "hsr_gazebo_move_base.launch"
    tree = ET.parse(launch_path)
    root = tree.getroot()

    assert root.tag == "launch"
    node_types = {(node.attrib.get("pkg"), node.attrib.get("type")) for node in root.iter("node")}
    assert ("gazebo_ros", "spawn_model") in node_types
    assert ("move_base", "move_base") in node_types
    assert ("vlmap_task_manager", "vlmap_task_manager_node") in node_types
    assert ("vlmap_task_manager", "vlmap_navigation_result_node") in node_types
    assert ("habitat_ros_bridge", "habitat_ros_bridge_node") in node_types

    params = {param.attrib.get("name"): param.attrib for param in root.iter("param")}
    assert params["robot_description"]["command"].startswith("$(find xacro)/xacro")
    assert "hsrb_description" in params["robot_description"]["command"]


def test_hsr_proxy_launch_is_valid_xml_and_dependency_free():
    launch_path = BRINGUP / "launch" / "hsr_proxy_gazebo_move_base.launch"
    tree = ET.parse(launch_path)
    root = tree.getroot()

    assert root.tag == "launch"
    node_types = {(node.attrib.get("pkg"), node.attrib.get("type")) for node in root.iter("node")}
    assert ("gazebo_ros", "spawn_model") in node_types
    assert ("move_base", "move_base") in node_types
    assert ("vlmap_task_manager", "vlmap_task_manager_node") in node_types
    assert ("habitat_ros_bridge", "habitat_ros_bridge_node") in node_types

    params = {param.attrib.get("name"): param.attrib for param in root.iter("param")}
    assert params["robot_description"]["textfile"] == "$(find vlmap_bringup)/urdf/hsr_proxy.urdf"


def test_hsr_proxy_urdf_exposes_hsr_frames_topics_and_sensors():
    urdf_path = BRINGUP / "urdf" / "hsr_proxy.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {link.attrib["name"] for link in root.iter("link")}
    assert "base_footprint" in links
    assert "base_link" in links
    assert "base_laser_link" in links
    assert "head_rgbd_sensor_rgb_frame" in links

    xml = urdf_path.read_text(encoding="utf-8")
    assert "/hsrb/command_velocity" in xml
    assert "/hsrb/base_scan" in xml
    assert "libgazebo_ros_diff_drive.so" in xml
    assert "libgazebo_ros_laser.so" in xml
    assert "libgazebo_ros_openni_kinect.so" in xml
    assert "rgb/image_raw" in xml
    assert "depth_registered/image_raw" in xml
    assert "rgb/camera_info" in xml


def test_hsr_move_base_profile_uses_hsr_frames_and_laser_topic():
    params = yaml.safe_load((BRINGUP / "config" / "hsr_move_base_params.yaml").read_text())

    assert params["global_costmap"]["robot_base_frame"] == "base_footprint"
    assert params["local_costmap"]["robot_base_frame"] == "base_footprint"
    assert params["local_costmap"]["obstacle_layer"]["hsr_base_scan"]["topic"] == "/hsrb/base_scan"
    assert params["DWAPlannerROS"]["holonomic_robot"] is False
    assert params["footprint"] != [[0.23, 0.17], [0.23, -0.17], [-0.23, -0.17], [-0.23, 0.17]]


def test_hsr_topic_contract_lists_navigation_and_camera_topics():
    topics = yaml.safe_load((BRINGUP / "config" / "hsr_topics.yaml").read_text())

    assert topics["frames"]["base"] == "base_footprint"
    assert topics["navigation"]["cmd_vel"] == "/hsrb/command_velocity"
    assert topics["navigation"]["base_scan"] == "/hsrb/base_scan"
    assert topics["perception"]["rgb_image"] == "/hsrb/head_rgbd_sensor/rgb/image_raw"
    assert topics["perception"]["depth_image"] == "/hsrb/head_rgbd_sensor/depth_registered/image_raw"
    assert topics["perception"]["camera_info"] == "/hsrb/head_rgbd_sensor/rgb/camera_info"


def test_hsr_stack_check_declares_required_hsr_package_and_topics():
    script_path = BRINGUP / "scripts" / "hsr_stack_check"
    loader = SourceFileLoader("hsr_stack_check", str(script_path))
    spec = importlib.util.spec_from_loader("hsr_stack_check", loader)
    assert spec is not None
    hsr_stack_check = importlib.util.module_from_spec(spec)
    sys.modules["hsr_stack_check"] = hsr_stack_check
    assert spec.loader is not None
    spec.loader.exec_module(hsr_stack_check)

    checks = {pkg.name: pkg for pkg in hsr_stack_check.PACKAGE_CHECKS}
    assert checks["hsrb_description"].required is True
    assert checks["hsrb_gazebo_launch"].required is False
    assert "/hsrb/base_scan" in hsr_stack_check.EXPECTED_TOPICS
    assert "/hsrb/head_rgbd_sensor/rgb/image_raw" in hsr_stack_check.EXPECTED_TOPICS
