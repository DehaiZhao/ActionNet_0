from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy import misc
from feature_extractor.feature_extractor import FeatureExtractor
import feature_extractor.utils as utils
import random
from tqdm import tqdm

path = '/home/cheer/video_test/corre/merge/jp_p_all'
label_name = 'label.txt'
ck_path = '/home/cheer/video_test/corre/model_si/jp_p_all/model.ckpt-30000'
num = 31
nums = 10
parts = ['T2', 'T3']
label_map = [[0,1,22,23,24], [2,5,16,19], [3,4,17,20], [6,8,9,13,21,28], [7,15], [10,29], [11], [26], [14,27], [12,18,25]]

def classification_placeholder_input(feature_extractor, image_path1, image_path2, logits_name, batch_size, num_classes):
  image_file1 = image_path1
  image_file2 = image_path2
  batch_image1 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)
  batch_image2 = np.zeros([batch_size, feature_extractor.image_size, feature_extractor.image_size, 3], dtype=np.float32)

  for i in range(batch_size):
    image1 = misc.imread(image_file1)
    image1 = misc.imresize(image1, (feature_extractor.image_size, feature_extractor.image_size))
    image1 = (image1/255.0).astype(dtype=np.float32)
    image1 -= 0.5
    image1 *= 2.0
    batch_image1[i] = image1

  for i in range(batch_size):
    image2 = misc.imread(image_file2)
    image2 = misc.imresize(image2, (feature_extractor.image_size, feature_extractor.image_size))
    image2 = (image2/255.0).astype(dtype=np.float32)
    image2 -= 0.5
    image2 *= 2.0
    batch_image2[i] = image2

  outputs = feature_extractor.feed_forward_batch([logits_name], batch_image1, batch_image2, fetch_images=True)

  predictions = np.squeeze(outputs[logits_name])
  predictions = np.argmax(predictions)
  return predictions

def convert_label(label_num):
  for i in range(len(label_map)):
    if label_num in label_map[i]:
      return i
      break
  return 0

def start(path):
  count = [0 for _ in range(num)]
  label_count = [0 for _ in range(num)]
  per_accuracy = [0 for _ in range(num)]
  total = 0
  with open(os.path.join(path, label_name)) as label_file:
    labels = label_file.readlines()

  labels = random.sample(labels, 1000)

  feature_extractor = FeatureExtractor(
    network_name='flownet_si',
    checkpoint_path=ck_path,
    batch_size=1,
    num_classes=num,
    preproc_func_name='flownet_si')
  feature_extractor.print_network_summary()

  for label in tqdm(labels):
    file_name = label.strip().split()[0]
    label_num = label.strip().split()[1]
    label_num = int(label_num)
    label_count[label_num] += 1
    image1 = os.path.join(path, parts[0], file_name)
    image2 = os.path.join(path, parts[1], file_name)
    clip_class = classification_placeholder_input(feature_extractor, image1, image2, 'Logits',1, 31)
    if clip_class == label_num:
      count[clip_class] += 1
      total += 1
  for i in range(len(label_count)):
    if label_count[i]:
      per_accuracy[i] = count[i]/label_count[i]
  print (label_count)
  print (count)
  print (per_accuracy)
  print (total)
  print (total/len(labels))

def main(path):
  count = [0 for _ in range(nums)]
  label_count = [0 for _ in range(nums)]
  per_accuracy = [0 for _ in range(nums)]
  TP = [0 for _ in range(nums)]
  TN = [0 for _ in range(nums)]
  FP = [0 for _ in range(nums)]
  FN = [0 for _ in range(nums)]
  recall = [0 for _ in range(nums)]
  precision = [0 for _ in range(nums)]
  f1 = [0 for _ in range(nums)]
  accuracy = [0 for _ in range(nums)]
  total = 0
  confusion = np.zeros((nums, nums), dtype = int)
  TN_matrix = np.zeros((nums, nums), dtype = int)
  with open(os.path.join(path, label_name)) as label_file:
    labels = label_file.readlines()

  labels = random.sample(labels, 1000)

  feature_extractor = FeatureExtractor(
    network_name='flownet_si',
    checkpoint_path=ck_path,
    batch_size=1,
    num_classes=num,
    preproc_func_name='flownet_si')
  feature_extractor.print_network_summary()

  for label in tqdm(labels):
    file_name = label.strip().split()[0]
    label_num = label.strip().split()[1]
    label_num = int(label_num)
    label_num = convert_label(label_num)
    label_count[label_num] += 1
    image1 = os.path.join(path, parts[0], file_name)
    image2 = os.path.join(path, parts[1], file_name)
    clip_class = classification_placeholder_input(feature_extractor, image1, image2, 'Logits',1, 31)
    clip_class = convert_label(clip_class)
    confusion[label_num][clip_class] += 1
    if clip_class == label_num:
      count[clip_class] += 1
      total += 1
  for i in range(len(label_count)):
    if label_count[i]:
      per_accuracy[i] = count[i]/label_count[i]
  for i in range(nums):
    TP[i] = confusion[i][i]
    FP[i] = np.sum(confusion[:,i]) - TP[i]
    FN[i] = np.sum(confusion[i]) - TP[i]
    TN[i] = len(labels) - TP[i] - FP[i] - FN[i]
    recall[i] = TP[i]/(TP[i]+FN[i])
    precision[i] = TP[i]/(TP[i]+FP[i])
    f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
    accuracy[i] = (TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]) 
  print (label_count)
  print (count)
  print (per_accuracy)
  print (total)
  print (total/len(labels))
  print (confusion)
  print ('TP:',TP)
  print ('FP:',FP)
  print ('FN:',FN)
  print ('TN:',TN)
  print ('precision:',precision)
  print ('recall:',recall)
  print ('f1:',f1)
  print ('accuracy:',accuracy)

if __name__ == '__main__':
  #start(path)
  main(path)
