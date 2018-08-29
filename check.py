import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm
dataset_path = '/home/cheer/video_test/corre/data'
parts1 = ['T0', 'T1']
parts2 = ['T4', 'T5']
label_name = 'label.txt'

def main():
  dir_list = os.listdir(dataset_path)
  for folder in dir_list:
    lsdir0 = os.listdir(os.path.join(dataset_path, folder, parts1[0]))
    lsdir1 = os.listdir(os.path.join(dataset_path, folder, parts1[1]))
    lsdir2 = os.listdir(os.path.join(dataset_path, folder, parts2[0]))
    lsdir3 = os.listdir(os.path.join(dataset_path, folder, parts2[1]))
    with open(os.path.join(dataset_path, folder, label_name)) as label_file:
      lines = label_file.readlines()
    if not len(lsdir0)==len(lines) and len(lsdir1)==len(lines) and len(lsdir2)==len(lines) and len(lsdir3)==len(lines):
      print 'error in folder', folder    

if __name__ == '__main__':
  main()
