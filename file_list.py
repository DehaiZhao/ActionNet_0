import random
import os
import cv2
from tqdm import tqdm
path = '/home/cheer/video_test/corre/data'

size = 299

def list_dir(path):
  lsdir = os.listdir(os.path.join(path, 't3'))
  for lf in lsdir:
    print 'rename:', os.path.join(path, 't3', lf.replace(' ', ''))
    os.rename(os.path.join(path, 't3', lf) , os.path.join(path, 't3', lf.replace(' ', '')))

def convert_dataset(path):
  if not os.path.exists(os.path.join(path, 'T2')):
    os.makedirs(os.path.join(path, 'T2'))
  if not os.path.exists(os.path.join(path, 'T3')):
    os.makedirs(os.path.join(path, 'T3'))
  with open(os.path.join(path, 'new_label.txt')) as label_file:
    label_list = label_file.readlines()
  for lines in label_list:
    file_name = lines.strip().split()[0]
    print file_name
    image_a = cv2.imread(os.path.join(path, 't4', file_name))
    image_a_resize = cv2.resize(image_a, (size,size))
    image_b = cv2.imread(os.path.join(path, 't5', file_name))
    image_b_resize = cv2.resize(image_b, (size,size))
    cv2.imwrite(os.path.join(path, 'T2', file_name), image_a_resize)
    cv2.imwrite(os.path.join(path, 'T3', file_name), image_b_resize)
  

if __name__ == '__main__':
  convert_dataset(path)
  #list_dir(path)
