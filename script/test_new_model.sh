protobuf_data_file_name='protobuf_fingerbendingmason1_36_all'
model_name='dbn'
model_file_name='model.test_dbn'
hidden_layers='50'
train_test_ratio='9'
learning_rate='.00005'
model='cnn'

# Train the model
python ../code/train.py \
	${protobuf_data_file_name} \
	${model_file_name} \
	${hidden_layers} \
	${train_test_ratio} \
	${learning_rate} \
	${model} \
	> ../log/train.log 2>&1

python ../code/predict.py offline_test \
	${model_name} ${model_file_name} ${hidden_layers} \
	${protobuf_data_file_name} ${train_test_ratio} | \
python ../code/visualize.py vis_result 
