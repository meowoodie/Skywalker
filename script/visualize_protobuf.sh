protobuf_path='../data/protobuf_fingerbendingmason1_72_binary_0'
# if the frame id is set to be -1, visualize all the frame in the protobuf file
frame_id='0'
# The size the each of the frame.
frame_size='72,72'
title='test_binary_72'

python ../code/visualize.py vis_data \
	${protobuf_path} ${frame_id} ${frame_size} ${title}
