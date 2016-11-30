protobuf_data_file_name='protobuf_fingerbendingmason1_36_binary_all'
#model_file_name='model.test_dbn'
#layers='50'
#train_test_ratio='9'
#learning_rate='.00005'
#model='dbn'
#pretrained='-1'

model_file_name='model.test_cnn'
layers='100,200#100'
train_test_ratio='9'
learning_rate='0.4'
model='cnn'
pretrained='-1'

# Train the model
python ../code/train.py \
	${protobuf_data_file_name} \
	${model_file_name} \
	${layers} \
	${train_test_ratio} \
	${learning_rate} \
	${model} \
	${pretrained} \
	> ../log/train.log 2>&1

# Test the model
python ../code/predict.py offline_test \
	${model} ${model_file_name} ${layers} \
	${protobuf_data_file_name} ${train_test_ratio} | \

python ../code/visualize.py vis_result 
