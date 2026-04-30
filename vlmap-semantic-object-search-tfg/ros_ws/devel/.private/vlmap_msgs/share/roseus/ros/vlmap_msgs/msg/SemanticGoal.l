;; Auto-generated. Do not edit!


(when (boundp 'vlmap_msgs::SemanticGoal)
  (if (not (find-package "VLMAP_MSGS"))
    (make-package "VLMAP_MSGS"))
  (shadow 'SemanticGoal (find-package "VLMAP_MSGS")))
(unless (find-package "VLMAP_MSGS::SEMANTICGOAL")
  (make-package "VLMAP_MSGS::SEMANTICGOAL"))

(in-package "ROS")
;;//! \htmlinclude SemanticGoal.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass vlmap_msgs::SemanticGoal
  :super ros::object
  :slots (_type _map_pose _room_id _object_class _metadata ))

(defmethod vlmap_msgs::SemanticGoal
  (:init
   (&key
    ((:type __type) "")
    ((:map_pose __map_pose) (instance geometry_msgs::PoseStamped :init))
    ((:room_id __room_id) "")
    ((:object_class __object_class) "")
    ((:metadata __metadata) "")
    )
   (send-super :init)
   (setq _type (string __type))
   (setq _map_pose __map_pose)
   (setq _room_id (string __room_id))
   (setq _object_class (string __object_class))
   (setq _metadata (string __metadata))
   self)
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:map_pose
   (&rest __map_pose)
   (if (keywordp (car __map_pose))
       (send* _map_pose __map_pose)
     (progn
       (if __map_pose (setq _map_pose (car __map_pose)))
       _map_pose)))
  (:room_id
   (&optional __room_id)
   (if __room_id (setq _room_id __room_id)) _room_id)
  (:object_class
   (&optional __object_class)
   (if __object_class (setq _object_class __object_class)) _object_class)
  (:metadata
   (&optional __metadata)
   (if __metadata (setq _metadata __metadata)) _metadata)
  (:serialization-length
   ()
   (+
    ;; string _type
    4 (length _type)
    ;; geometry_msgs/PoseStamped _map_pose
    (send _map_pose :serialization-length)
    ;; string _room_id
    4 (length _room_id)
    ;; string _object_class
    4 (length _object_class)
    ;; string _metadata
    4 (length _metadata)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _type
       (write-long (length _type) s) (princ _type s)
     ;; geometry_msgs/PoseStamped _map_pose
       (send _map_pose :serialize s)
     ;; string _room_id
       (write-long (length _room_id) s) (princ _room_id s)
     ;; string _object_class
       (write-long (length _object_class) s) (princ _object_class s)
     ;; string _metadata
       (write-long (length _metadata) s) (princ _metadata s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _type
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _type (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; geometry_msgs/PoseStamped _map_pose
     (send _map_pose :deserialize buf ptr-) (incf ptr- (send _map_pose :serialization-length))
   ;; string _room_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _room_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _object_class
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _object_class (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _metadata
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _metadata (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get vlmap_msgs::SemanticGoal :md5sum-) "5ad7085f3396999b2c42c4f499c94bef")
(setf (get vlmap_msgs::SemanticGoal :datatype-) "vlmap_msgs/SemanticGoal")
(setf (get vlmap_msgs::SemanticGoal :definition-)
      "# Goal expressed in semantic terms (mirror of src/tfg_nav_contracts/SemanticGoal).
#
# Field semantics:
#   type          one of: ROOM, APPROACH, INSPECT, VERIFY (string for clarity)
#   map_pose      goal pose in the /map frame
#   room_id       room label assigned by the room labeler (\"kitchen\", ...)
#   object_class  target category for verification (\"sofa\", \"bottle\", ...)
#   metadata      free-form JSON string with extra context (priors, etc.)
string type
geometry_msgs/PoseStamped map_pose
string room_id
string object_class
string metadata

================================================================================
MSG: geometry_msgs/PoseStamped
# A Pose with reference coordinate frame and timestamp
Header header
Pose pose

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :vlmap_msgs/SemanticGoal "5ad7085f3396999b2c42c4f499c94bef")


