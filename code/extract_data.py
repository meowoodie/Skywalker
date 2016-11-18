import sys
import cv2
import arrow
import imageio
from skimage.feature import canny
from skimage.filters import threshold_otsu
import lib.ultrasound_with_gesture_pb2 as ultra 
from heapq import nsmallest

#########################
# Preprocess the raw data
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def parse_gesture_timestamp(t):
    time_info = t.strip().split('-')
    sec_milli_string = time_info.pop()
    # Millisecond
    milli_time = sec_milli_string[-3:]
    milli_time = 0.001 * float(milli_time)
    # Second
    sec_time = sec_milli_string[:-3]
    time_info.append(sec_time)
    format_time_info = []
    for t in time_info:
        if len(t) == 1:
            t = '0' + t
        if len(t) == 0:
            t = '00'
        format_time_info.append(t)
    arrow_time_string = '%s-%s-%s %s:%s:%s' % tuple(format_time_info)
    arrow_time_format = 'YYYY-MM-DD HH:mm:ss'
    arrow_time = arrow.get(arrow_time_string, arrow_time_format)
    return float("{:.3f}".format(arrow_time.timestamp + milli_time))

def preprocess_glove_data(glove_data):
    new_glove_data = []
    for item in glove_data:
        item[0] = parse_gesture_timestamp(item[0])
        item = map(float, item)
        new_glove_data.append(item)
    return new_glove_data

def find_k_nearest(array, target, k):
    return nsmallest(k, array, key=lambda x: abs(x[0]-target))
    
##############
# File handler
def write2protobuf(protobuf_name, protobuf_obj):
    # Write protocol buffer into local file
    print >> sys.stderr, '[%s] Start writing buffer content to file' % arrow.now()
    buf_str = protobuf_obj.SerializeToString()
    with open(protobuf_name, 'w') as buf_file:
        buf_file.write(buf_str)

#################################
# Process each frame of the video
def binarizing(image, threshold=-1):
    bias = 10.0
    if threshold == -1:
        threshold = threshold_otsu(image) + bias
    binary = image > threshold
    return binary

def downscaling(image, proportion):
    return cv2.resize(image, (0,0), fx=proportion, fy=proportion)

def preprocess_frame(image, process_steps):
    for step in process_steps:
        if step[0] == 'binary':
            print >> sys.stderr, '[%s] Binarizing...' % arrow.now()
            image = binarizing(image, float(step[1]))
        elif step[0] == 'downscale':
            print >> sys.stderr, '[%s] downscaling...' % arrow.now()
            image = downscaling(image, float(step[1]))
        elif step[0] == 'canny':
            print >> sys.stderr, '[%s] canny...' % arrow.now()
            image = canny(image, float(step[1]))
    return image

#######################################
# Read data file name from command line
video_name = sys.argv[1]      # input file name
glove_name = sys.argv[2]      # input file name
protobuf_name = sys.argv[3]   # output file name
# Properties of the video
start_time  = float(sys.argv[4]) # Unixtime of the start time
start_frame = int(sys.argv[5])   # The start frame
frame_rate  = float(sys.argv[6]) # The frame rate of the video
# Properties of the protobuf
buf_size    = int(sys.argv[7]) # the size of each of the buffers
buf_num     = int(sys.argv[8]) # if you want to process all the data, set -1
proc_mode   = sys.argv[9].split('#')
#video cropping
top_crop, bot_crop, left_crop, right_crop = map(int, sys.argv[10].strip().split(','))
allow_error = 0.02

# Parse the processing mode
process_steps = map(lambda step: step.split(','), proc_mode)
print >> sys.stderr, '[%s] the processing mode including:' % arrow.now()
for step in process_steps:
    print >> sys.stderr, 'Mode: %s, Parameters: %s' % tuple(step) 


##########
# The main

# Read video data from file
cap = imageio.get_reader(video_name)
frame_number = cap.get_length()
# Read glove data from file
raw_gestures = []
with open(glove_name) as data_file:
    for line in data_file:
        raw_gestures.append(line.strip().split('\t'))
gestures = preprocess_glove_data(raw_gestures)

# Initialize the protocolbuffer
video_buf = ultra.UltrasoundVideo()
video_buf.name = protobuf_name + '_0'

# Read each image from the video
print >> sys.stderr, '[%s] Start reading Image data from the video...' % arrow.now()
j = 0 # valid frame index
last_time = 0.0
cur_time  = start_time
for i, im in enumerate(cap):
    print >> sys.stderr, '[%s] Frame %i' % (arrow.now(), i)
    if i < start_frame:
        print >> sys.stderr, '[%s] Invalid frame, discarded.' % arrow.now()
        continue
    # Read image
    frame = cap.get_data(i)

    # crop image
    if top_crop != -1 and bot_crop != -1 and left_crop != -1 and right_crop != -1:
        frame_rows = frame.shape[0]
        frame_cols = frame.shape[1]
        frame = frame[top_crop:frame_rows-bot_crop, left_crop:frame_cols-right_crop]

    # Grey processing each image in the video 
    print >> sys.stderr, '[%s] Gray processing...' % arrow.now()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:    
        proc_img = preprocess_frame(gray_image, process_steps)
    except Exception as e:
        print >> sys.stderr, '[%s] Preprocess failed: %s' % (arrow.now(), e)
        continue
    print >> sys.stderr, '[%s] Current size of the frame: %s * %s' % (arrow.now(), proc_img.shape[0], proc_img.shape[1])
    # Calculate the timestamp of the next frame
    last_time = cur_time
    cur_time += 1.0 / frame_rate

    # Fill in the buffer with the corresponding gesture data
    print >> sys.stderr, '[%s] Finding matched gesture...' % arrow.now()
    best_mateched = find_k_nearest(gestures, cur_time, 1)[0]
    print >> sys.stderr, '[%s] current frame timestamp:\t%s' % (arrow.now(), cur_time)
    print >> sys.stderr, '[%s] matched gesture timestamp:\t%s' % (arrow.now(), best_mateched[0])
    if abs(best_mateched[0] - cur_time) > allow_error:
        print >> sys.stderr, '[%s] Match failed, error:%s' % (arrow.now(), abs(best_mateched[0] - cur_time))
        continue

    # Fill in the buffer with each image
    print >> sys.stderr, '[%s] Fill in the buffer...' % arrow.now()

    image_buf   = video_buf.ultrasound_images.add()
    gesture_buf = image_buf.gesture
    gesture_buf.wrist_roll   = best_mateched[1]
    gesture_buf.wrist_pitch  = best_mateched[2]
    gesture_buf.wrist_yaw    = best_mateched[3]
    gesture_buf.hand_roll    = best_mateched[4]
    gesture_buf.hand_pitch   = best_mateched[5]
    gesture_buf.hand_yaw     = best_mateched[6]
    gesture_buf.thumb_bend   = best_mateched[7]
    gesture_buf.index_bend   = best_mateched[8]
    gesture_buf.middle_bend  = best_mateched[9]
    gesture_buf.ring_bend    = best_mateched[10]
    gesture_buf.pinky_bend   = best_mateched[11]
    gesture_buf.thumb_press  = best_mateched[12]
    gesture_buf.index_press  = best_mateched[13]
    gesture_buf.middle_press = best_mateched[14]
    gesture_buf.ring_press   = best_mateched[15]
    gesture_buf.pinky_press  = best_mateched[16]

    image_buf.rows = proc_img.shape[0]
    image_buf.cols = proc_img.shape[1]
    image_buf.timestamp = cur_time
    for x in range(proc_img.shape[0]):
        for y in range(proc_img.shape[1]):
            image_buf.pixels.append(int(proc_img[x][y]))

    # Write to file
    if buf_size != -1 and (j+1) % buf_size == 0:
        write2protobuf(protobuf_name + '_' + str(j/buf_size), video_buf)
        # Initialize the protocolbuffer
        video_buf = ultra.UltrasoundVideo()
        video_buf.name = video_name + '_' + str(j/buf_size)
    
    if buf_size != -1 and buf_num != -1 and (j/buf_size) >= buf_num:
        break
    
    j += 1

if buf_size == -1:
    write2protobuf(protobuf_name + '_all', video_buf)

