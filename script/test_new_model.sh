protobuf_data_file_name='protobuf_finger_bending_mason_1_ds36_first1000'
model_file_name='model.test_dbn'
hidden_layers='10000'
train_test_ratio='9'

# Train the model
python ../code/train.py \
	${protobuf_data_file_name} \
	${model_file_name} \
	${hidden_layers} \
	${train_test_ratio} \
	> ../log/train.log 2>&1

# Run test_main() in skywalkder.py
# python ../code/contrast.py \
# 	${model_file_name} \
# 	${test_data_file_name} \
# 	${feature_size} ${label_size} ${hidden_layers} \
# 	> ../results/${contrast_result_file_name} \
# 	2> ../log/contrast.log

# Plot the contrast results
# cat ../results/${contrast_result_file_name} | python ../code/plot/plot_contrast.py
