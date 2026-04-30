; Auto-generated. Do not edit!


(cl:in-package vlmap_msgs-msg)


;//! \htmlinclude SemanticGoal.msg.html

(cl:defclass <SemanticGoal> (roslisp-msg-protocol:ros-message)
  ((type
    :reader type
    :initarg :type
    :type cl:string
    :initform "")
   (map_pose
    :reader map_pose
    :initarg :map_pose
    :type geometry_msgs-msg:PoseStamped
    :initform (cl:make-instance 'geometry_msgs-msg:PoseStamped))
   (room_id
    :reader room_id
    :initarg :room_id
    :type cl:string
    :initform "")
   (object_class
    :reader object_class
    :initarg :object_class
    :type cl:string
    :initform "")
   (metadata
    :reader metadata
    :initarg :metadata
    :type cl:string
    :initform ""))
)

(cl:defclass SemanticGoal (<SemanticGoal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SemanticGoal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SemanticGoal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vlmap_msgs-msg:<SemanticGoal> is deprecated: use vlmap_msgs-msg:SemanticGoal instead.")))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <SemanticGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-msg:type-val is deprecated.  Use vlmap_msgs-msg:type instead.")
  (type m))

(cl:ensure-generic-function 'map_pose-val :lambda-list '(m))
(cl:defmethod map_pose-val ((m <SemanticGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-msg:map_pose-val is deprecated.  Use vlmap_msgs-msg:map_pose instead.")
  (map_pose m))

(cl:ensure-generic-function 'room_id-val :lambda-list '(m))
(cl:defmethod room_id-val ((m <SemanticGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-msg:room_id-val is deprecated.  Use vlmap_msgs-msg:room_id instead.")
  (room_id m))

(cl:ensure-generic-function 'object_class-val :lambda-list '(m))
(cl:defmethod object_class-val ((m <SemanticGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-msg:object_class-val is deprecated.  Use vlmap_msgs-msg:object_class instead.")
  (object_class m))

(cl:ensure-generic-function 'metadata-val :lambda-list '(m))
(cl:defmethod metadata-val ((m <SemanticGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-msg:metadata-val is deprecated.  Use vlmap_msgs-msg:metadata instead.")
  (metadata m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SemanticGoal>) ostream)
  "Serializes a message object of type '<SemanticGoal>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'type))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'type))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'map_pose) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'room_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'room_id))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'object_class))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'object_class))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'metadata))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'metadata))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SemanticGoal>) istream)
  "Deserializes a message object of type '<SemanticGoal>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'type) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'map_pose) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'room_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'room_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'object_class) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'object_class) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'metadata) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'metadata) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SemanticGoal>)))
  "Returns string type for a message object of type '<SemanticGoal>"
  "vlmap_msgs/SemanticGoal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SemanticGoal)))
  "Returns string type for a message object of type 'SemanticGoal"
  "vlmap_msgs/SemanticGoal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SemanticGoal>)))
  "Returns md5sum for a message object of type '<SemanticGoal>"
  "5ad7085f3396999b2c42c4f499c94bef")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SemanticGoal)))
  "Returns md5sum for a message object of type 'SemanticGoal"
  "5ad7085f3396999b2c42c4f499c94bef")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SemanticGoal>)))
  "Returns full string definition for message of type '<SemanticGoal>"
  (cl:format cl:nil "# Goal expressed in semantic terms (mirror of src/tfg_nav_contracts/SemanticGoal).~%#~%# Field semantics:~%#   type          one of: ROOM, APPROACH, INSPECT, VERIFY (string for clarity)~%#   map_pose      goal pose in the /map frame~%#   room_id       room label assigned by the room labeler (\"kitchen\", ...)~%#   object_class  target category for verification (\"sofa\", \"bottle\", ...)~%#   metadata      free-form JSON string with extra context (priors, etc.)~%string type~%geometry_msgs/PoseStamped map_pose~%string room_id~%string object_class~%string metadata~%~%================================================================================~%MSG: geometry_msgs/PoseStamped~%# A Pose with reference coordinate frame and timestamp~%Header header~%Pose pose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SemanticGoal)))
  "Returns full string definition for message of type 'SemanticGoal"
  (cl:format cl:nil "# Goal expressed in semantic terms (mirror of src/tfg_nav_contracts/SemanticGoal).~%#~%# Field semantics:~%#   type          one of: ROOM, APPROACH, INSPECT, VERIFY (string for clarity)~%#   map_pose      goal pose in the /map frame~%#   room_id       room label assigned by the room labeler (\"kitchen\", ...)~%#   object_class  target category for verification (\"sofa\", \"bottle\", ...)~%#   metadata      free-form JSON string with extra context (priors, etc.)~%string type~%geometry_msgs/PoseStamped map_pose~%string room_id~%string object_class~%string metadata~%~%================================================================================~%MSG: geometry_msgs/PoseStamped~%# A Pose with reference coordinate frame and timestamp~%Header header~%Pose pose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SemanticGoal>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'type))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'map_pose))
     4 (cl:length (cl:slot-value msg 'room_id))
     4 (cl:length (cl:slot-value msg 'object_class))
     4 (cl:length (cl:slot-value msg 'metadata))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SemanticGoal>))
  "Converts a ROS message object to a list"
  (cl:list 'SemanticGoal
    (cl:cons ':type (type msg))
    (cl:cons ':map_pose (map_pose msg))
    (cl:cons ':room_id (room_id msg))
    (cl:cons ':object_class (object_class msg))
    (cl:cons ':metadata (metadata msg))
))
