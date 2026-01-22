# the default GPU is the one with the least memory 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

Rain100L_train = datasets/Rain100L/train
Rain100L_val = datasets/Rain100L/test
Rain200L_train = datasets/Rain200L/train
Rain200L_val = datasets/Rain200L/test
Rain200H_train = datasets/Rain200H/train
Rain200H_val = datasets/Rain200H/test
Rain800_train = datasets/Rain800/train
Rain800_val = datasets/Rain800/test

DEV = $(shell nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print NR-1,$$1}' | sort -k2 -n | awk '{print $$1}' | head -1)

DATASET=Rain100L
NAME=${DATASET}
train_path=${${DATASET}_train}
val_path=${${DATASET}_val}
EXPERIMENT=experiment

train_test:
	CUDA_VISIBLE_DEVICES=${DEV} python3 train.py \
		--result_dir ${EXPERIMENT} \
		--name ${NAME} \
		--train_path ${train_path} \
		--val_path ${val_path}

	CUDA_VISIBLE_DEVICES=${DEV} python3 test.py \
		--result_dir outputs_before \
		--name ${NAME} \
		--test_path ${val_path}/input \
		--resume ${EXPERIMENT}/${NAME}/net1.pth
	cd statistic && matlab -r "compute_metrics('../${val_path}/target/', '../outputs_before/${NAME}/'); exit"

train:
	CUDA_VISIBLE_DEVICES=${DEV} python3 train.py \
		--result_dir ${EXPERIMENT} \
		--name ${NAME} \
		--train_path ${train_path} \
		--val_path ${val_path}

test:
	CUDA_VISIBLE_DEVICES=${DEV} python3 test.py \
		--result_dir outputs_before \
		--name ${NAME} \
		--test_path ${val_path}/input \
		--resume ${EXPERIMENT}/${NAME}/net1.pth
	cd statistic && matlab -r "compute_metrics('../${val_path}/target/', '../outputs_before/${NAME}/'); exit"