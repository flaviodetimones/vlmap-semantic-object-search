;; Auto-generated. Do not edit!


(when (boundp 'vlmap_msgs::ResolveRoom)
  (if (not (find-package "VLMAP_MSGS"))
    (make-package "VLMAP_MSGS"))
  (shadow 'ResolveRoom (find-package "VLMAP_MSGS")))
(unless (find-package "VLMAP_MSGS::RESOLVEROOM")
  (make-package "VLMAP_MSGS::RESOLVEROOM"))
(unless (find-package "VLMAP_MSGS::RESOLVEROOMREQUEST")
  (make-package "VLMAP_MSGS::RESOLVEROOMREQUEST"))
(unless (find-package "VLMAP_MSGS::RESOLVEROOMRESPONSE")
  (make-package "VLMAP_MSGS::RESOLVEROOMRESPONSE"))

(in-package "ROS")





(defclass vlmap_msgs::ResolveRoomRequest
  :super ros::object
  :slots (_room ))

(defmethod vlmap_msgs::ResolveRoomRequest
  (:init
   (&key
    ((:room __room) "")
    )
   (send-super :init)
   (setq _room (string __room))
   self)
  (:room
   (&optional __room)
   (if __room (setq _room __room)) _room)
  (:serialization-length
   ()
   (+
    ;; string _room
    4 (length _room)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _room
       (write-long (length _room) s) (princ _room s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _room
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _room (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass vlmap_msgs::ResolveRoomResponse
  :super ros::object
  :slots (_found _room_id _x _y ))

(defmethod vlmap_msgs::ResolveRoomResponse
  (:init
   (&key
    ((:found __found) nil)
    ((:room_id __room_id) "")
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    )
   (send-super :init)
   (setq _found __found)
   (setq _room_id (string __room_id))
   (setq _x (float __x))
   (setq _y (float __y))
   self)
  (:found
   (&optional (__found :null))
   (if (not (eq __found :null)) (setq _found __found)) _found)
  (:room_id
   (&optional __room_id)
   (if __room_id (setq _room_id __room_id)) _room_id)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:serialization-length
   ()
   (+
    ;; bool _found
    1
    ;; string _room_id
    4 (length _room_id)
    ;; float32 _x
    4
    ;; float32 _y
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _found
       (if _found (write-byte -1 s) (write-byte 0 s))
     ;; string _room_id
       (write-long (length _room_id) s) (princ _room_id s)
     ;; float32 _x
       (sys::poke _x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y
       (sys::poke _y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _found
     (setq _found (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _room_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _room_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _x
     (setq _x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y
     (setq _y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass vlmap_msgs::ResolveRoom
  :super ros::object
  :slots ())

(setf (get vlmap_msgs::ResolveRoom :md5sum-) "db7b5f5b60012129daeee10be8237c87")
(setf (get vlmap_msgs::ResolveRoom :datatype-) "vlmap_msgs/ResolveRoom")
(setf (get vlmap_msgs::ResolveRoom :request) vlmap_msgs::ResolveRoomRequest)
(setf (get vlmap_msgs::ResolveRoom :response) vlmap_msgs::ResolveRoomResponse)

(defmethod vlmap_msgs::ResolveRoomRequest
  (:response () (instance vlmap_msgs::ResolveRoomResponse :init)))

(setf (get vlmap_msgs::ResolveRoomRequest :md5sum-) "db7b5f5b60012129daeee10be8237c87")
(setf (get vlmap_msgs::ResolveRoomRequest :datatype-) "vlmap_msgs/ResolveRoomRequest")
(setf (get vlmap_msgs::ResolveRoomRequest :definition-)
      "# Resolve a room label into a concrete 2D goal in the /map frame.
#
# Request:
#   room  semantic room label or alias, e.g. \"kitchen\", \"office\".
# Response:
#   found    whether the room could be resolved
#   room_id  exact room label used internally by the provider
#   x        x coordinate in the /map frame (metres)
#   y        y coordinate in the /map frame (metres)
string room
---
bool found
string room_id
float32 x
float32 y

")

(setf (get vlmap_msgs::ResolveRoomResponse :md5sum-) "db7b5f5b60012129daeee10be8237c87")
(setf (get vlmap_msgs::ResolveRoomResponse :datatype-) "vlmap_msgs/ResolveRoomResponse")
(setf (get vlmap_msgs::ResolveRoomResponse :definition-)
      "# Resolve a room label into a concrete 2D goal in the /map frame.
#
# Request:
#   room  semantic room label or alias, e.g. \"kitchen\", \"office\".
# Response:
#   found    whether the room could be resolved
#   room_id  exact room label used internally by the provider
#   x        x coordinate in the /map frame (metres)
#   y        y coordinate in the /map frame (metres)
string room
---
bool found
string room_id
float32 x
float32 y

")



(provide :vlmap_msgs/ResolveRoom "db7b5f5b60012129daeee10be8237c87")


