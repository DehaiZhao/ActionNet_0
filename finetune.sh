#!/bin/bash
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
#PRETRAINED_CHECKPOINT_DIR=/home/cheer/video_test/classifier/partial_clip/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=flownet_si
TRAINSET_NAME=jp_p_all
#EVALUATIONSET_NAME=python_0_0

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/cheer/video_test/corre/model_si/${TRAINSET_NAME}

# Where the dataset is saved to.
TRAINSET_DIR=/home/cheer/video_test/corre/merge/${TRAINSET_NAME}
#EVALUATIONSET_DIR=/home/cheer/video_test/corre/data/${EVALUATIONSET_NAME}

# Fine-tune only the new layers for 20000 steps.

python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=30000 \
  --batch_size=16 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.00004

