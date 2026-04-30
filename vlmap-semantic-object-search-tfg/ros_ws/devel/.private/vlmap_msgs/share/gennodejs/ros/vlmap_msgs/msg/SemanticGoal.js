// Auto-generated. Do not edit!

// (in-package vlmap_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class SemanticGoal {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.type = null;
      this.map_pose = null;
      this.room_id = null;
      this.object_class = null;
      this.metadata = null;
    }
    else {
      if (initObj.hasOwnProperty('type')) {
        this.type = initObj.type
      }
      else {
        this.type = '';
      }
      if (initObj.hasOwnProperty('map_pose')) {
        this.map_pose = initObj.map_pose
      }
      else {
        this.map_pose = new geometry_msgs.msg.PoseStamped();
      }
      if (initObj.hasOwnProperty('room_id')) {
        this.room_id = initObj.room_id
      }
      else {
        this.room_id = '';
      }
      if (initObj.hasOwnProperty('object_class')) {
        this.object_class = initObj.object_class
      }
      else {
        this.object_class = '';
      }
      if (initObj.hasOwnProperty('metadata')) {
        this.metadata = initObj.metadata
      }
      else {
        this.metadata = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SemanticGoal
    // Serialize message field [type]
    bufferOffset = _serializer.string(obj.type, buffer, bufferOffset);
    // Serialize message field [map_pose]
    bufferOffset = geometry_msgs.msg.PoseStamped.serialize(obj.map_pose, buffer, bufferOffset);
    // Serialize message field [room_id]
    bufferOffset = _serializer.string(obj.room_id, buffer, bufferOffset);
    // Serialize message field [object_class]
    bufferOffset = _serializer.string(obj.object_class, buffer, bufferOffset);
    // Serialize message field [metadata]
    bufferOffset = _serializer.string(obj.metadata, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SemanticGoal
    let len;
    let data = new SemanticGoal(null);
    // Deserialize message field [type]
    data.type = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [map_pose]
    data.map_pose = geometry_msgs.msg.PoseStamped.deserialize(buffer, bufferOffset);
    // Deserialize message field [room_id]
    data.room_id = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [object_class]
    data.object_class = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [metadata]
    data.metadata = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.type);
    length += geometry_msgs.msg.PoseStamped.getMessageSize(object.map_pose);
    length += _getByteLength(object.room_id);
    length += _getByteLength(object.object_class);
    length += _getByteLength(object.metadata);
    return length + 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'vlmap_msgs/SemanticGoal';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5ad7085f3396999b2c42c4f499c94bef';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SemanticGoal(null);
    if (msg.type !== undefined) {
      resolved.type = msg.type;
    }
    else {
      resolved.type = ''
    }

    if (msg.map_pose !== undefined) {
      resolved.map_pose = geometry_msgs.msg.PoseStamped.Resolve(msg.map_pose)
    }
    else {
      resolved.map_pose = new geometry_msgs.msg.PoseStamped()
    }

    if (msg.room_id !== undefined) {
      resolved.room_id = msg.room_id;
    }
    else {
      resolved.room_id = ''
    }

    if (msg.object_class !== undefined) {
      resolved.object_class = msg.object_class;
    }
    else {
      resolved.object_class = ''
    }

    if (msg.metadata !== undefined) {
      resolved.metadata = msg.metadata;
    }
    else {
      resolved.metadata = ''
    }

    return resolved;
    }
};

module.exports = SemanticGoal;
