import os
import sys
import cv2
import shutil
import numpy as np
import random
from tqdm import tqdm
dataset_path = '/home/cheer/video_test/corre/data'
output_path = '/home/cheer/video_test/corre/random_sample'
parts = ['t0', 't1']

def create_folder(output_path, folder):
  for part in parts:
    if not os.path.exists(os.path.join(output_path, folder, part)):
      print 'create folder {}'.format(folder)
      os.makedirs(os.path.join(output_path, folder, part))
    else:
      print 'folder {} exist'.format(folder)

def main(dataset_path):
  dir_list = os.listdir(dataset_path)
  for folder in tqdm(dir_list):
    create_folder(output_path, folder)
    if os.path.exists(os.path.join(dataset_path, folder, parts[0])):
      image_list = os.listdir(os.path.join(dataset_path, folder, parts[0]))
    else:
      image_list = []
    if len(image_list):
      image_list = random.sample(image_list, len(image_list)/5)
    for image in image_list:
      shutil.copyfile(os.path.join(dataset_path, folder, parts[0], image), os.path.join(output_path, folder, parts[0], image))
      shutil.copyfile(os.path.join(dataset_path, folder, parts[1], image), os.path.join(output_path, folder, parts[1], image))

if __name__ == '__main__':
  main(dataset_path)
