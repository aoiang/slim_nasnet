#!/usr/bin/env bash
set -e
set -x

DATASET_NAME=cifar10 # imagenet or cifar10
DATASET_DIR=/home/wwu12/yiyang/cifar10
TRAIN_DIR=/home/wwu12/yiyang/32_batch_log/
export CUDA_VISIBLE_DEVICES=0
MODEL=nasnet_cifar
OPTIMIZER=momentum
LEARNING_RATE_DECAY_TYPE=cosine
NUM_EPOCHS_PER_DECAY=600
TRAIN_BATCH_SIZE=32
LEARNING_RATE=0.025
EVAL_INTERVAL_SECS=600
INFO_WORKSPACE=eval_info
IMAGE_SIZE=32

if [ ! -d "$TRAIN_DIR" ]; then
  echo "${TRAIN_DIR} does not exsit!"
  exit
fi

if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi

python train_image_classifier.py --dataset_dir=${DATASET_DIR} \
--train_dir=${TRAIN_DIR} \
--dataset_split_name=train  \
--dataset_name=${DATASET_NAME}  \
--model_name=${MODEL} \
--optimizer=${OPTIMIZER}  \
--learning_rate_decay_type=${LEARNING_RATE_DECAY_TYPE} \
--num_epochs_per_decay=${NUM_EPOCHS_PER_DECAY} \
--batch_size=${TRAIN_BATCH_SIZE} \
--train_image_size=${IMAGE_SIZE} \
--learning_rate=${LEARNING_RATE} > ${INFO_WORKSPACE}/train_info.txt 2>&1 &

python eval_image_classifier.py --alsologtostderr \
--checkpoint_path=${TRAIN_DIR} \
--dataset_name=${DATASET_NAME} \
--dataset_split_name=test \
--dataset_dir=${DATASET_DIR} \
--model_name=${MODEL} \
--preprocessing_name=cifarnet >  ${INFO_WORKSPACE}/eval_info.txt 2>&1 &

