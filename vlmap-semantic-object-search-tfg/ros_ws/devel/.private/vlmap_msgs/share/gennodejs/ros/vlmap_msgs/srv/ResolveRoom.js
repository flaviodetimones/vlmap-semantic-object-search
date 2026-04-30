// Auto-generated. Do not edit!

// (in-package vlmap_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class ResolveRoomRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.room = null;
    }
    else {
      if (initObj.hasOwnProperty('room')) {
        this.room = initObj.room
      }
      else {
        this.room = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResolveRoomRequest
    // Serialize message field [room]
    bufferOffset = _serializer.string(obj.room, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResolveRoomRequest
    let len;
    let data = new ResolveRoomRequest(null);
    // Deserialize message field [room]
    data.room = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.room);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'vlmap_msgs/ResolveRoomRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e497569192cccb82020c3a5c262721b9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Resolve a room label into a concrete 2D goal in the /map frame.
    #
    # Request:
    #   room  semantic room label or alias, e.g. "kitchen", "office".
    # Response:
    #   found    whether the room could be resolved
    #   room_id  exact room label used internally by the provider
    #   x        x coordinate in the /map frame (metres)
    #   y        y coordinate in the /map frame (metres)
    string room
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResolveRoomRequest(null);
    if (msg.room !== undefined) {
      resolved.room = msg.room;
    }
    else {
      resolved.room = ''
    }

    return resolved;
    }
};

class ResolveRoomResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.found = null;
      this.room_id = null;
      this.x = null;
      this.y = null;
    }
    else {
      if (initObj.hasOwnProperty('found')) {
        this.found = initObj.found
      }
      else {
        this.found = false;
      }
      if (initObj.hasOwnProperty('room_id')) {
        this.room_id = initObj.room_id
      }
      else {
        this.room_id = '';
      }
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = 0.0;
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResolveRoomResponse
    // Serialize message field [found]
    bufferOffset = _serializer.bool(obj.found, buffer, bufferOffset);
    // Serialize message field [room_id]
    bufferOffset = _serializer.string(obj.room_id, buffer, bufferOffset);
    // Serialize message field [x]
    bufferOffset = _serializer.float32(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = _serializer.float32(obj.y, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResolveRoomResponse
    let len;
    let data = new ResolveRoomResponse(null);
    // Deserialize message field [found]
    data.found = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [room_id]
    data.room_id = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [x]
    data.x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.room_id);
    return length + 13;
  }

  static datatype() {
    // Returns string type for a service object
    return 'vlmap_msgs/ResolveRoomResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'be97407ed30065a5f9e6923ae6c8fa60';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool found
    string room_id
    float32 x
    float32 y
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResolveRoomResponse(null);
    if (msg.found !== undefined) {
      resolved.found = msg.found;
    }
    else {
      resolved.found = false
    }

    if (msg.room_id !== undefined) {
      resolved.room_id = msg.room_id;
    }
    else {
      resolved.room_id = ''
    }

    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = 0.0
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = 0.0
    }

    return resolved;
    }
};

module.exports = {
  Request: ResolveRoomRequest,
  Response: ResolveRoomResponse,
  md5sum() { return 'db7b5f5b60012129daeee10be8237c87'; },
  datatype() { return 'vlmap_msgs/ResolveRoom'; }
};
