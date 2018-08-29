#!/bin/bash
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
#PRETRAINED_CHECKPOINT_DIR=/home/cheer/video_test/classifier/partial_clip/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=flownet_s
TRAINSET_NAME=python_0_0
EVALUATIONSET_NAME=java_0_0

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/cheer/video_test/corre/model/${TRAINSET_NAME}

# Where the dataset is saved to.
TRAINSET_DIR=/home/cheer/video_test/corre/data/${TRAINSET_NAME}
EVALUATIONSET_DIR=/home/cheer/video_test/corre/data/${EVALUATIONSET_NAME}

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=validation \
  --dataset_dir=${EVALUATIONSET_DIR} \
  --model_name=${MODEL_NAME}
