protobuf_data_file_name='protobuf_fingerbendingmason1_36_binary_all'
#model_file_name='model.test_dbn'
#layers='50'
#train_test_ratio='9'
#learning_rate='.00005'
#model='dbn'
#pretrained='-1'

conv1=$1
conv2=$2
hid=$3
lr=$4

model_file_name="cnn.c1_${conv1}.c2_${conv2}.h1_${hid}.lr_${lr}"
layers="${conv1},${conv2}#${hid}"
train_test_ratio='9'
learning_rate=${lr}
model='cnn'
pretrained='-1'
#pretrained='model.test_cnn'

# Train the model
python ../code/train.py \
	${protobuf_data_file_name} \
	${model_file_name} \
	${layers} \
	${train_test_ratio} \
	${learning_rate} \
	${model} \
	${pretrained} \
	> ../log/train.c1_${conv1}.c2_${conv2}.h1_${hid}.lr_${lr}.log 2>&1

# Test the model
python ../code/predict.py offline_test \
	${model} ${model_file_name} ${layers} \
	${protobuf_data_file_name} ${train_test_ratio} | \

python ../code/visualize.py vis_result ${model_file_name} 
