protobuf_data_file_name='protobuf_finger_bending_mason_no1_ds36_all'
model_name='dbn'
model_file_name='model.test_dbn'
hidden_layers='1000'
train_test_ratio='9'

# Train the model
python ../code/train.py \
	${protobuf_data_file_name} \
	${model_file_name} \
	${hidden_layers} \
	${train_test_ratio} \
	> ../log/train.log 2>&1

python ../code/predict.py offline_test \
	${model_name} ${model_file_name} ${hidden_layers} \
	${protobuf_data_file_name} ${train_test_ratio} > contrast.txt

cat contrast.txt | python ../code/visualize.py vis_result 
