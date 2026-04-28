"""yoloe_verifier — YOLOE-based visual verification as a ROS service.

Sprint 1: stub. Sprint 2 will:
  - subscribe to /camera/color/image_raw (sensor_msgs/Image)
  - expose /yoloe/check (custom srv) returning (found, bbox, score)
  - delegate to vlmaps.utils.yoloe_utils.get_session inside tfg-sim via roslibpy
    (or, alternatively, ship the YOLOE weights into tfg-ros — TBD in Sprint 2).
"""
