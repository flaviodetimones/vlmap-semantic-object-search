
(cl:in-package :asdf)

(defsystem "vlmap_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "QueryRoom" :depends-on ("_package_QueryRoom"))
    (:file "_package_QueryRoom" :depends-on ("_package"))
    (:file "ResolveRoom" :depends-on ("_package_ResolveRoom"))
    (:file "_package_ResolveRoom" :depends-on ("_package"))
  ))