; Auto-generated. Do not edit!


(cl:in-package vlmap_msgs-srv)


;//! \htmlinclude QueryRoom-request.msg.html

(cl:defclass <QueryRoom-request> (roslisp-msg-protocol:ros-message)
  ((category
    :reader category
    :initarg :category
    :type cl:string
    :initform ""))
)

(cl:defclass QueryRoom-request (<QueryRoom-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QueryRoom-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QueryRoom-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vlmap_msgs-srv:<QueryRoom-request> is deprecated: use vlmap_msgs-srv:QueryRoom-request instead.")))

(cl:ensure-generic-function 'category-val :lambda-list '(m))
(cl:defmethod category-val ((m <QueryRoom-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:category-val is deprecated.  Use vlmap_msgs-srv:category instead.")
  (category m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QueryRoom-request>) ostream)
  "Serializes a message object of type '<QueryRoom-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'category))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'category))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QueryRoom-request>) istream)
  "Deserializes a message object of type '<QueryRoom-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'category) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'category) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QueryRoom-request>)))
  "Returns string type for a service object of type '<QueryRoom-request>"
  "vlmap_msgs/QueryRoomRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QueryRoom-request)))
  "Returns string type for a service object of type 'QueryRoom-request"
  "vlmap_msgs/QueryRoomRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QueryRoom-request>)))
  "Returns md5sum for a message object of type '<QueryRoom-request>"
  "dd7014894137ba2b5c36b11fca65c3bb")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QueryRoom-request)))
  "Returns md5sum for a message object of type 'QueryRoom-request"
  "dd7014894137ba2b5c36b11fca65c3bb")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QueryRoom-request>)))
  "Returns full string definition for message of type '<QueryRoom-request>"
  (cl:format cl:nil "# Ask the semantic server which rooms most likely contain a category.~%#~%# Request:~%#   category  free text label, e.g. \"sofa\", \"bottle\".~%# Response:~%#   rooms     ordered list of room ids, best first.~%#   scores    matching list of priors / scores for each room.~%string category~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QueryRoom-request)))
  "Returns full string definition for message of type 'QueryRoom-request"
  (cl:format cl:nil "# Ask the semantic server which rooms most likely contain a category.~%#~%# Request:~%#   category  free text label, e.g. \"sofa\", \"bottle\".~%# Response:~%#   rooms     ordered list of room ids, best first.~%#   scores    matching list of priors / scores for each room.~%string category~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QueryRoom-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'category))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QueryRoom-request>))
  "Converts a ROS message object to a list"
  (cl:list 'QueryRoom-request
    (cl:cons ':category (category msg))
))
;//! \htmlinclude QueryRoom-response.msg.html

(cl:defclass <QueryRoom-response> (roslisp-msg-protocol:ros-message)
  ((rooms
    :reader rooms
    :initarg :rooms
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element ""))
   (scores
    :reader scores
    :initarg :scores
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass QueryRoom-response (<QueryRoom-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QueryRoom-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QueryRoom-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vlmap_msgs-srv:<QueryRoom-response> is deprecated: use vlmap_msgs-srv:QueryRoom-response instead.")))

(cl:ensure-generic-function 'rooms-val :lambda-list '(m))
(cl:defmethod rooms-val ((m <QueryRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:rooms-val is deprecated.  Use vlmap_msgs-srv:rooms instead.")
  (rooms m))

(cl:ensure-generic-function 'scores-val :lambda-list '(m))
(cl:defmethod scores-val ((m <QueryRoom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vlmap_msgs-srv:scores-val is deprecated.  Use vlmap_msgs-srv:scores instead.")
  (scores m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QueryRoom-response>) ostream)
  "Serializes a message object of type '<QueryRoom-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'rooms))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'rooms))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'scores))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'scores))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QueryRoom-response>) istream)
  "Deserializes a message object of type '<QueryRoom-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'rooms) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'rooms)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'scores) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'scores)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QueryRoom-response>)))
  "Returns string type for a service object of type '<QueryRoom-response>"
  "vlmap_msgs/QueryRoomResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QueryRoom-response)))
  "Returns string type for a service object of type 'QueryRoom-response"
  "vlmap_msgs/QueryRoomResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QueryRoom-response>)))
  "Returns md5sum for a message object of type '<QueryRoom-response>"
  "dd7014894137ba2b5c36b11fca65c3bb")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QueryRoom-response)))
  "Returns md5sum for a message object of type 'QueryRoom-response"
  "dd7014894137ba2b5c36b11fca65c3bb")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QueryRoom-response>)))
  "Returns full string definition for message of type '<QueryRoom-response>"
  (cl:format cl:nil "string[] rooms~%float32[] scores~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QueryRoom-response)))
  "Returns full string definition for message of type 'QueryRoom-response"
  (cl:format cl:nil "string[] rooms~%float32[] scores~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QueryRoom-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'rooms) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'scores) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QueryRoom-response>))
  "Converts a ROS message object to a list"
  (cl:list 'QueryRoom-response
    (cl:cons ':rooms (rooms msg))
    (cl:cons ':scores (scores msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'QueryRoom)))
  'QueryRoom-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'QueryRoom)))
  'QueryRoom-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QueryRoom)))
  "Returns string type for a service object of type '<QueryRoom>"
  "vlmap_msgs/QueryRoom")