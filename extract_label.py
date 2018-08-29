import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm
dataset_path = '/home/cheer/video_test/corre/data'
partial_path = '/home/cheer/video_test/corre/diff_data'
parts0 = ['t0', 't1']
parts1 = ['T4', 'T5']
parts2 = ['T0', 'T1']
label_name = 'label.txt'

def create_folder(folder):
  for part in parts1:
    if not os.path.exists(os.path.join(dataset_path, folder, part)):
      print 'create folder {}'.format(folder)
      os.makedirs(os.path.join(dataset_path, folder, part))
    else:
      print 'folder {} exist'.format(folder)

def rm_folder():
  dir_list = os.listdir(dataset_path)
  for folder in dir_list:
    try:
      os.remove(os.path.join(dataset_path, folder, parts1[0], folder + '00000.jpg'))
    except:
      print 'eroor in folder', folder

def copy0():
  dir_list = os.listdir(dataset_path)
  for folder in tqdm(dir_list):
    try:
      shutil.copyfile(os.path.join(dataset_path, folder, parts2[0], folder + '_00000.jpg'), os.path.join(dataset_path, folder, parts1[0], folder + '_00000.jpg'))
      shutil.copyfile(os.path.join(dataset_path, folder, parts2[1], folder + '_00000.jpg'), os.path.join(dataset_path, folder, parts1[1], folder + '_00000.jpg'))
    except:
      print 'eroor in folder', folder
      
def main():
  dir_list = os.listdir(dataset_path)
  for folder in dir_list:
    create_folder(folder)
    lsdir = os.listdir(os.path.join(partial_path, folder, 't0'))
    lsdir.sort()
    with open(os.path.join(dataset_path, folder, label_name)) as label_file:
      lines = label_file.readlines()
    for line in tqdm(lines):
      image_name = line.strip().split()[0]
      i = int(os.path.splitext(line.strip().split()[0])[0].split('_')[-1])
      try:
        image1 = cv2.imread(os.path.join(partial_path, folder, parts0[0], lsdir[i]))
        image2 = cv2.imread(os.path.join(partial_path, folder, parts0[1], lsdir[i]))
        image1 = cv2.resize(image1, (299, 299))
        image2 = cv2.resize(image2, (299, 299))
        cv2.imwrite(os.path.join(dataset_path, folder, parts1[0], image_name), image1)
        cv2.imwrite(os.path.join(dataset_path, folder, parts1[1], image_name), image2)
      except:
        print '**********error in folder', folder, image_name

if __name__ == '__main__':
  #rm_folder()
  main()
  #copy0()
