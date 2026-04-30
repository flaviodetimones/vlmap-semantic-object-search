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

class QueryRoomRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.category = null;
    }
    else {
      if (initObj.hasOwnProperty('category')) {
        this.category = initObj.category
      }
      else {
        this.category = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QueryRoomRequest
    // Serialize message field [category]
    bufferOffset = _serializer.string(obj.category, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QueryRoomRequest
    let len;
    let data = new QueryRoomRequest(null);
    // Deserialize message field [category]
    data.category = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.category);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'vlmap_msgs/QueryRoomRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7397d5563c5bad7f8044755b463842e8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Ask the semantic server which rooms most likely contain a category.
    #
    # Request:
    #   category  free text label, e.g. "sofa", "bottle".
    # Response:
    #   rooms     ordered list of room ids, best first.
    #   scores    matching list of priors / scores for each room.
    string category
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QueryRoomRequest(null);
    if (msg.category !== undefined) {
      resolved.category = msg.category;
    }
    else {
      resolved.category = ''
    }

    return resolved;
    }
};

class QueryRoomResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.rooms = null;
      this.scores = null;
    }
    else {
      if (initObj.hasOwnProperty('rooms')) {
        this.rooms = initObj.rooms
      }
      else {
        this.rooms = [];
      }
      if (initObj.hasOwnProperty('scores')) {
        this.scores = initObj.scores
      }
      else {
        this.scores = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QueryRoomResponse
    // Serialize message field [rooms]
    bufferOffset = _arraySerializer.string(obj.rooms, buffer, bufferOffset, null);
    // Serialize message field [scores]
    bufferOffset = _arraySerializer.float32(obj.scores, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QueryRoomResponse
    let len;
    let data = new QueryRoomResponse(null);
    // Deserialize message field [rooms]
    data.rooms = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [scores]
    data.scores = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.rooms.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    length += 4 * object.scores.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'vlmap_msgs/QueryRoomResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ed3772818fc622ee82ba797ed36e139f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[] rooms
    float32[] scores
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QueryRoomResponse(null);
    if (msg.rooms !== undefined) {
      resolved.rooms = msg.rooms;
    }
    else {
      resolved.rooms = []
    }

    if (msg.scores !== undefined) {
      resolved.scores = msg.scores;
    }
    else {
      resolved.scores = []
    }

    return resolved;
    }
};

module.exports = {
  Request: QueryRoomRequest,
  Response: QueryRoomResponse,
  md5sum() { return 'dd7014894137ba2b5c36b11fca65c3bb'; },
  datatype() { return 'vlmap_msgs/QueryRoom'; }
};
