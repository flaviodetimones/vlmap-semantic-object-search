# Goal expressed in semantic terms (mirror of src/tfg_nav_contracts/SemanticGoal).
#
# Field semantics:
#   type          one of: ROOM, APPROACH, INSPECT, VERIFY (string for clarity)
#   map_pose      goal pose in the /map frame
#   room_id       room label assigned by the room labeler ("kitchen", ...)
#   object_class  target category for verification ("sofa", "bottle", ...)
#   metadata      free-form JSON string with extra context (priors, etc.)
string type
geometry_msgs/PoseStamped map_pose
string room_id
string object_class
string metadata
