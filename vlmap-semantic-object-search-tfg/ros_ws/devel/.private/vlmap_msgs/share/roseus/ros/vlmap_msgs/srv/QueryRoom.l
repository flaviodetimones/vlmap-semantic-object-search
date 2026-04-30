;; Auto-generated. Do not edit!


(when (boundp 'vlmap_msgs::QueryRoom)
  (if (not (find-package "VLMAP_MSGS"))
    (make-package "VLMAP_MSGS"))
  (shadow 'QueryRoom (find-package "VLMAP_MSGS")))
(unless (find-package "VLMAP_MSGS::QUERYROOM")
  (make-package "VLMAP_MSGS::QUERYROOM"))
(unless (find-package "VLMAP_MSGS::QUERYROOMREQUEST")
  (make-package "VLMAP_MSGS::QUERYROOMREQUEST"))
(unless (find-package "VLMAP_MSGS::QUERYROOMRESPONSE")
  (make-package "VLMAP_MSGS::QUERYROOMRESPONSE"))

(in-package "ROS")





(defclass vlmap_msgs::QueryRoomRequest
  :super ros::object
  :slots (_category ))

(defmethod vlmap_msgs::QueryRoomRequest
  (:init
   (&key
    ((:category __category) "")
    )
   (send-super :init)
   (setq _category (string __category))
   self)
  (:category
   (&optional __category)
   (if __category (setq _category __category)) _category)
  (:serialization-length
   ()
   (+
    ;; string _category
    4 (length _category)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _category
       (write-long (length _category) s) (princ _category s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _category
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _category (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass vlmap_msgs::QueryRoomResponse
  :super ros::object
  :slots (_rooms _scores ))

(defmethod vlmap_msgs::QueryRoomResponse
  (:init
   (&key
    ((:rooms __rooms) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:scores __scores) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _rooms __rooms)
   (setq _scores __scores)
   self)
  (:rooms
   (&optional __rooms)
   (if __rooms (setq _rooms __rooms)) _rooms)
  (:scores
   (&optional __scores)
   (if __scores (setq _scores __scores)) _scores)
  (:serialization-length
   ()
   (+
    ;; string[] _rooms
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _rooms)) 4
    ;; float32[] _scores
    (* 4    (length _scores)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _rooms
     (write-long (length _rooms) s)
     (dolist (elem _rooms)
       (write-long (length elem) s) (princ elem s)
       )
     ;; float32[] _scores
     (write-long (length _scores) s)
     (dotimes (i (length _scores))
       (sys::poke (elt _scores i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _rooms
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _rooms (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _rooms i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; float32[] _scores
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _scores (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _scores i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass vlmap_msgs::QueryRoom
  :super ros::object
  :slots ())

(setf (get vlmap_msgs::QueryRoom :md5sum-) "dd7014894137ba2b5c36b11fca65c3bb")
(setf (get vlmap_msgs::QueryRoom :datatype-) "vlmap_msgs/QueryRoom")
(setf (get vlmap_msgs::QueryRoom :request) vlmap_msgs::QueryRoomRequest)
(setf (get vlmap_msgs::QueryRoom :response) vlmap_msgs::QueryRoomResponse)

(defmethod vlmap_msgs::QueryRoomRequest
  (:response () (instance vlmap_msgs::QueryRoomResponse :init)))

(setf (get vlmap_msgs::QueryRoomRequest :md5sum-) "dd7014894137ba2b5c36b11fca65c3bb")
(setf (get vlmap_msgs::QueryRoomRequest :datatype-) "vlmap_msgs/QueryRoomRequest")
(setf (get vlmap_msgs::QueryRoomRequest :definition-)
      "# Ask the semantic server which rooms most likely contain a category.
#
# Request:
#   category  free text label, e.g. \"sofa\", \"bottle\".
# Response:
#   rooms     ordered list of room ids, best first.
#   scores    matching list of priors / scores for each room.
string category
---
string[] rooms
float32[] scores

")

(setf (get vlmap_msgs::QueryRoomResponse :md5sum-) "dd7014894137ba2b5c36b11fca65c3bb")
(setf (get vlmap_msgs::QueryRoomResponse :datatype-) "vlmap_msgs/QueryRoomResponse")
(setf (get vlmap_msgs::QueryRoomResponse :definition-)
      "# Ask the semantic server which rooms most likely contain a category.
#
# Request:
#   category  free text label, e.g. \"sofa\", \"bottle\".
# Response:
#   rooms     ordered list of room ids, best first.
#   scores    matching list of priors / scores for each room.
string category
---
string[] rooms
float32[] scores

")



(provide :vlmap_msgs/QueryRoom "dd7014894137ba2b5c36b11fca65c3bb")


