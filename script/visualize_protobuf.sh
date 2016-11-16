protobuf_path='../data/protobuf_fingerbendingmason1_36_all'
# if the frame id is set to be -1, visualize all the frame in the protobuf file
frame_id='-1'
# The size the each of the frame.
frame_size='36,36'
title='test_36'

python ../code/visualize.py vis_data \
	${protobuf_path} ${frame_id} ${frame_size} ${title}
