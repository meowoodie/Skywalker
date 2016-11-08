protobuf_path='../data/protobuf_finger_bending_mason_1_ds360_canny_1'
# if the frame id is set to be -1, visualize all the frame in the protobuf file
frame_id='0'
# The size the each of the frame.
frame_size='360,360'
title='test_canny'

python ../code/visualize.py vis_data \
	${protobuf_path} ${frame_id} ${frame_size} ${title}
