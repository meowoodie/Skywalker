import sys
import numpy as np
import ultrasound_with_gesture_pb2 as ultra
from tempfile import TemporaryFile

def read_data_from_protobuf(file_path):
    # Init the ultra object
    video_buf = ultra.UltrasoundVideo()
    with open(file_path, 'rb') as buf_file:
        video_buf.ParseFromString(buf_file.read())
    
    name = video_buf.name
    
    # Init the output variable
    image_list     = []
    gesture_list   = []
    timestamp_List = []

    for ultra_image in video_buf.ultrasound_images:
        # Gesture
        #if not ultra_image.HasField('gesture'):
        #    print >> sys.stderr, 'Missing gesture, discard this frame'
        #    continue
        gesture    = [ -1 for i in range(10) ]
        gesture[0] = ultra_image.gesture.thumb_bend
        gesture[1] = ultra_image.gesture.index_bend
        gesture[2] = ultra_image.gesture.middle_bend
        gesture[3] = ultra_image.gesture.ring_bend
        gesture[4] = ultra_image.gesture.pinky_bend
        gesture[5] = ultra_image.gesture.thumb_press
        gesture[6] = ultra_image.gesture.index_press
        gesture[7] = ultra_image.gesture.middle_press
        gesture[8] = ultra_image.gesture.ring_press
        gesture[9] = ultra_image.gesture.pinky_press
        gesture_list.append(gesture)
        # Basic information
        rows       = ultra_image.rows
        cols       = ultra_image.cols
        timestamp  = ultra_image.timestamp
        timestamp_List.append(timestamp)
        # Pixels
        pixels     = []
        for pixel in ultra_image.pixels:
            pixels.append(pixel)
        pixels = np.array(pixels).reshape(rows, cols)
        image_list.append(pixels)

    return name, np.array(image_list), np.array(gesture_list), np.array(timestamp_List)

# TODO
def convert_protobuf2numpy(protobuf_file_path, numpy_file_name):
    _, image_list, gesture_list, _ = read_data_from_protobuf(protobuf_file_path)
    outfile = TemporaryFile()
    np.savez(outfile, features=image_list, labels=gesture_list)


if __name__ == '__main__':
    file_path = '../../data/protobuf_finger_bending_mason_1_0'
    # name, image_list, gesture_list, timestamp_list = read_data_from_protobuf(file_path)
    # print 'Name:\t%s' % name
    # print 'Len of image list:\t%s' % len(image_list)
    # print 'Len of gesture list:\t%s' % len(gesture_list)
    # print 'Len of timestamp list:\t%s' % len(timestamp_list)
    # print 'A sample of image:'
    # print image_list[2]
    # print 'Image shape: %s * %s' % (image_list[0].shape[0], image_list[0].shape[1])
    # print 'A sample of gesture:'
    # print gesture_list[1]
    # print 'A sample of timestamp:'
    # print timestamp_list[0]

    convert_protobuf2numpy(file_path, 'test_numpy')
