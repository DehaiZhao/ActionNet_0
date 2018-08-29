from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

path = '/home/cheer/video_test/corre/merge/java_all'
data_path = '/home/cheer/video_test/corre/data'
label_name = 'label.txt'
num = 31
label_map = [[0,1,22,23,24], [2,5,16,19], [3,4,17,20], [6,8,9,13,21,28], [7,15], [10,29], [11], [26], [14,27], [12,18,25]]

def start():
  with open(os.path.join(path, label_name)) as label_file:
    labels = label_file.readlines()
  label_count = [0 for _ in range(num)]
  count = []
  for label in tqdm(labels):
    label_num = label.strip().split()[1]
    label_num = int(label_num)
    count.append(label_num)
    label_count[label_num] += 1
  print (label_count)
  plt.hist(count, num)
  plt.show()

def convert_label(label_num):
  for i in range(len(label_map)):
    if label_num in label_map[i]:
      return i
      break
  return 0

def count_raw():
  lsdir = os.listdir(data_path)
  lsdir.sort()
  count1 = 0
  count2 = 0
  image_count = 0
  for i in range(25):
    print(lsdir[i])
    images = os.listdir(os.path.join(data_path, lsdir[i], 't0'))
    count1 += len(images)
  print (count1)
  for i in range(25, 50):
    print(lsdir[i])
    images = os.listdir(os.path.join(data_path, lsdir[i], 't0'))
    count2 += len(images)
  print (count2)
  print (count1 + count2)

def spe_count():
  with open(os.path.join(path, label_name)) as label_file:
    labels = label_file.readlines()
  label_count = [0 for _ in range(num)]
  label_o = 0
  count = 0
  for label in tqdm(labels):
    label_num = label.strip().split()[1]
    label_num = int(label_num)
    label_num = convert_label(label_num)
    if label_o == 8 and label_num == 7:
      count += 1
    label_o = label_num
  print (count)

if __name__ == '__main__':
  #start()
  #count_raw()
  spe_count()
