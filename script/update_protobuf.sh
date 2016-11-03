SRC_DIR='..'
DST_DIR='../code'
PROTOBUF_FILE_NAME='ultrasound_with_gesture.proto'
protoc -I=${SRC_DIR} --python_out=${DST_DIR} ${SRC_DIR}/${PROTOBUF_FILE_NAME}
