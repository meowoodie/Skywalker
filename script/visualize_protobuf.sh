protobuf_path='../data/protobuf_finger_bending_mason_1_ds36_first1000'
# if the frame id is set to be -1, visualize all the frame in the protobuf file
frame_id='1'
# The size the each of the frame.
frame_size='36,36'
title='test'

python ../code/visualize.py vis_data \
	${protobuf_path} ${frame_id} ${frame_size} ${title}
