; Auto-generated. Do not edit!


(cl:in-package vlmap_msgs-srv)


;//! \htmlinclude ResolveRoom-request.msg.html

(cl:defclass <ResolveRoom-request> (roslisp-msg-protocol:ros-message)
  ((room
    :reader room
    :initarg :room
    :type cl:string
    :initform ""))
)

(cl:defclass ResolveRoom-request (<ResolveRoom-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResolveRoom-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResolveRoom-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vlmap_msgs-srv:<ResolveRoom-request> is deprecated: use vlmap_msgs-srv:ResolveRoom-request instead.")))

(cl:ensure-generic-function 'room-val :lambda-list '(m))
(cl:defmethod room-val ((m <ResolveRoom-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:room-val is deprecated.  Use vlmap_msgs-srv:room instead.")
  (room m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResolveRoom-request>) ostream)
  "Serializes a message object of type '<ResolveRoom-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'room))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'room))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResolveRoom-request>) istream)
  "Deserializes a message object of type '<ResolveRoom-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'room) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'room) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResolveRoom-request>)))
  "Returns string type for a service object of type '<ResolveRoom-request>"
  "vlmap_msgs/ResolveRoomRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResolveRoom-request)))
  "Returns string type for a service object of type 'ResolveRoom-request"
  "vlmap_msgs/ResolveRoomRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResolveRoom-request>)))
  "Returns md5sum for a message object of type '<ResolveRoom-request>"
  "db7b5f5b60012129daeee10be8237c87")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResolveRoom-request)))
  "Returns md5sum for a message object of type 'ResolveRoom-request"
  "db7b5f5b60012129daeee10be8237c87")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResolveRoom-request>)))
  "Returns full string definition for message of type '<ResolveRoom-request>"
  (cl:format cl:nil "# Resolve a room label into a concrete 2D goal in the /map frame.~%#~%# Request:~%#   room  semantic room label or alias, e.g. \"kitchen\", \"office\".~%# Response:~%#   found    whether the room could be resolved~%#   room_id  exact room label used internally by the provider~%#   x        x coordinate in the /map frame (metres)~%#   y        y coordinate in the /map frame (metres)~%string room~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResolveRoom-request)))
  "Returns full string definition for message of type 'ResolveRoom-request"
  (cl:format cl:nil "# Resolve a room label into a concrete 2D goal in the /map frame.~%#~%# Request:~%#   room  semantic room label or alias, e.g. \"kitchen\", \"office\".~%# Response:~%#   found    whether the room could be resolved~%#   room_id  exact room label used internally by the provider~%#   x        x coordinate in the /map frame (metres)~%#   y        y coordinate in the /map frame (metres)~%string room~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResolveRoom-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'room))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResolveRoom-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ResolveRoom-request
    (cl:cons ':room (room msg))
))
;//! \htmlinclude ResolveRoom-response.msg.html

(cl:defclass <ResolveRoom-response> (roslisp-msg-protocol:ros-message)
  ((found
    :reader found
    :initarg :found
    :type cl:boolean
    :initform cl:nil)
   (room_id
    :reader room_id
    :initarg :room_id
    :type cl:string
    :initform "")
   (x
    :reader x
    :initarg :x
    :type cl:float
    :initform 0.0)
   (y
    :reader y
    :initarg :y
    :type cl:float
    :initform 0.0))
)

(cl:defclass ResolveRoom-response (<ResolveRoom-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResolveRoom-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResolveRoom-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vlmap_msgs-srv:<ResolveRoom-response> is deprecated: use vlmap_msgs-srv:ResolveRoom-response instead.")))

(cl:ensure-generic-function 'found-val :lambda-list '(m))
(cl:defmethod found-val ((m <ResolveRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:found-val is deprecated.  Use vlmap_msgs-srv:found instead.")
  (found m))

(cl:ensure-generic-function 'room_id-val :lambda-list '(m))
(cl:defmethod room_id-val ((m <ResolveRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:room_id-val is deprecated.  Use vlmap_msgs-srv:room_id instead.")
  (room_id m))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <ResolveRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:x-val is deprecated.  Use vlmap_msgs-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <ResolveRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:y-val is deprecated.  Use vlmap_msgs-srv:y instead.")
  (y m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResolveRoom-response>) ostream)
  "Serializes a message object of type '<ResolveRoom-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'found) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'room_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'room_id))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResolveRoom-response>) istream)
  "Deserializes a message object of type '<ResolveRoom-response>"
    (cl:setf (cl:slot-value msg 'found) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'room_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'room_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResolveRoom-response>)))
  "Returns string type for a service object of type '<ResolveRoom-response>"
  "vlmap_msgs/ResolveRoomResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResolveRoom-response)))
  "Returns string type for a service object of type 'ResolveRoom-response"
  "vlmap_msgs/ResolveRoomResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResolveRoom-response>)))
  "Returns md5sum for a message object of type '<ResolveRoom-response>"
  "db7b5f5b60012129daeee10be8237c87")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResolveRoom-response)))
  "Returns md5sum for a message object of type 'ResolveRoom-response"
  "db7b5f5b60012129daeee10be8237c87")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResolveRoom-response>)))
  "Returns full string definition for message of type '<ResolveRoom-response>"
  (cl:format cl:nil "bool found~%string room_id~%float32 x~%float32 y~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResolveRoom-response)))
  "Returns full string definition for message of type 'ResolveRoom-response"
  (cl:format cl:nil "bool found~%string room_id~%float32 x~%float32 y~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResolveRoom-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'room_id))
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResolveRoom-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ResolveRoom-response
    (cl:cons ':found (found msg))
    (cl:cons ':room_id (room_id msg))
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ResolveRoom)))
  'ResolveRoom-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ResolveRoom)))
  'ResolveRoom-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResolveRoom)))
  "Returns string type for a service object of type '<ResolveRoom>"
  "vlmap_msgs/ResolveRoom")