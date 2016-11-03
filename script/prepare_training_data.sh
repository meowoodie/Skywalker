VIDEO_PATH='../data/FingerBending4GB.mp4'
GLOVE_PATH='../data/finger_bending_mason_1.txt'
PROTO_BUF_PATH='../data/protobuf_finger_bending_mason_1_ds36'
# The timestamp of the start frame
start_time='1476463978.558'
# The start frame is the first valid frame which is decided by observing manually
start_frame='143'
# The frame rate of the video
frame_rate='30'
# The size of each of the buffer
buf_size='1000'
# The number of the buffer that you want to output, 
# if you want to process all the data, set buf_num='-1'
buf_num='1' 
# Preprocessing mode
# Each processing step is divided by '#', e.g. [step1]#[step2]#...
# The before comma part is the name of processing, including: binary, canny, downscale
# The after comma part is the parameter of processing, including:
# - binary: threshold, if set -1 means automatically find threshold
# - downscale: the proportion of the downscaling
# - canny: sigma of the canny
proc_mode='downscale,0.05#binary,-1'


python ../code/extract_data.py ${VIDEO_PATH} ${GLOVE_PATH} ${PROTO_BUF_PATH} \
	${start_time} ${start_frame} ${frame_rate} \
	${buf_size} ${buf_num} ${proc_mode} 2> ../log/extract_data.log



