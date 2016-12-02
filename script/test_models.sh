#!/bin/bash
index=(1 2 3 4 5 6)
conv1=("110" "100" "160" "120")
conv2=("75" "50" "40" "40")
hid=("200" "225" "250" "300")
learning_rate=("0.5" "0.55" "0.45" "0.6" "0.4")

for i in ${index[@]}
do
	for h in ${hid[@]}
	do
		for lr in ${learning_rate[@]}
		do
			echo "conv1:${conv1[$i]}, conv2:${conv2[$i]}, hid:${h}, lr:${lr}"
			sh test_new_model.sh ${conv1[$i]} ${conv2[$i]} ${h} ${lr}
		done
	done
done
