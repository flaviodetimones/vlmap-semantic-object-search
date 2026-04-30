
(cl:in-package :asdf)

(defsystem "vlmap_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "SemanticGoal" :depends-on ("_package_SemanticGoal"))
    (:file "_package_SemanticGoal" :depends-on ("_package"))
  ))