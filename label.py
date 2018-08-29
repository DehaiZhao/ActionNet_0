import random
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
path = '/home/cheer/video_test/partial_clip'
dataset = '/home/cheer/video_test/corre/data'
parts = ['t4', 't5']

def create_folder(dataset, parts):
  for part in parts:
    if not os.path.exists(os.path.join(dataset, part)):
      print 'create folder {}'.format(part)
      os.makedirs(os.path.join(dataset, part))
    else:
      print 'folder exist'

def read_i(dataset):
  if os.path.isfile(os.path.join(dataset, 'new_label.txt')):
    with open(os.path.join(dataset, 'new_label.txt'), 'r') as label:
      lines = label.readlines()
      if len(lines) > 0:
        i = int(os.path.splitext(lines[-1].strip().split()[0])[0])
        return i+1
      else:
        return 0
  else:
    return 0
  

def main(path):
  cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  lsdir = os.listdir(path)
  lsdir.sort()
  i = read_i(dataset)
  while i < (len(lsdir) - 1):
    image1 = cv2.imread(os.path.join(path, lsdir[i]))
    image2 = cv2.imread(os.path.join(path, lsdir[i+1]))
    image1 = cv2.resize(image1, (200, 200))
    image2 = cv2.resize(image2, (200, 200))
    image = np.hstack((image1, image2))
    print i
    print lsdir[i]
    print lsdir[i+1]
    cv2.imshow('image', image)
    k = cv2.waitKey(1)
    if k == ord('q'):
      break
    classnumber = input('class number:')
    if classnumber == 99:
      i += 1
      continue
    elif classnumber == 88:
      continue
    else:
      with open(os.path.join(dataset, 'new_label.txt'), 'a') as label:
        label.write('{:05d}.jpg {}\n'.format(i, classnumber))
        shutil.copyfile(os.path.join(path, lsdir[i]), os.path.join(dataset, parts[0], '{:05d}.jpg'.format(i)))
        shutil.copyfile(os.path.join(path, lsdir[i+1]), os.path.join(dataset, parts[1], '{:05d}.jpg'.format(i)))
        i += 1
        continue

def rnm(path):
  namelist = []
  lsdir = os.listdir(path)
  for filename in tqdm(lsdir):
    name = os.path.splitext(filename)[0]
    name = '%05d' % int(name)
    name = str(name) + '.jpg'
    os.rename(os.path.join(path, filename), os.path.join(path, name))


if __name__ == '__main__':
  create_folder(dataset, parts)
  main(path)
  #rnm(path)
