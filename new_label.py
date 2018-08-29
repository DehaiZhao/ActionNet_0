import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm
dataset_path = '/home/cheer/video_test/corre/random_sample'
folder_name = 'java_4_4'
label_file = 'label.txt'
parts = ['T0', 'T1']

def create_folder(dataset_path):
  print folder_name
  for part in parts:
    if not os.path.exists(os.path.join(dataset_path, folder_name, part)):
      print 'create folder {}'.format(part)
      os.makedirs(os.path.join(dataset_path, folder_name, part))
    else:
      print 'folder {} exist'.format(part)

def read_i(dataset_path):
  if os.path.isfile(os.path.join(dataset_path, folder_name, label_file)):
    with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
      lines = label.readlines()
      if len(lines) > 0:
        i = int(os.path.splitext(lines[-1].strip().split()[0])[0].split('_')[-1])
        return i+1
      else:
        return 0
  else:
    return 0

def read_label(dataset_path):
  if os.path.isfile(os.path.join(dataset_path, folder_name, label_file)):
    with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
      lines = label.readlines()
      if len(lines) > 0:
        num = int(lines[-1].strip().split()[1])
        return num
      else:
        return 999
  else:
    return 999

def delete_file(i):
  with open(os.path.join(dataset_path, folder_name, label_file), 'r') as label:
    lines = label.readlines()
  lines.pop(-1)
  with open(os.path.join(dataset_path, folder_name, label_file), 'w') as label:
    label.writelines(lines)
  os.remove(os.path.join(dataset_path, folder_name, parts[0], folder_name + '_{:05d}.jpg'.format(i)))
  os.remove(os.path.join(dataset_path, folder_name, parts[1], folder_name + '_{:05d}.jpg'.format(i)))
  
def main(dataset_path):
  cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  lsdir = os.listdir(os.path.join(dataset_path, folder_name, 't0'))
  lsdir.sort()
  i = read_i(dataset_path)
  class_num = 999
  num_buffer = []
  while i < (len(lsdir)):
    sys.stdout.write('reading image' + str(lsdir[i]) + ' ' + str(i) + '/' + str(len(lsdir)) + '\r')
    sys.stdout.flush()
    image1 = cv2.imread(os.path.join(dataset_path, folder_name, 't0', lsdir[i]))
    image2 = cv2.imread(os.path.join(dataset_path, folder_name, 't1', lsdir[i]))
    image1 = cv2.resize(image1, (299, 299))
    image2 = cv2.resize(image2, (299, 299))
    image3 = np.zeros_like(image1)
    image3 = cv2.resize(image3, (5, 299))
    image = np.hstack((image1, image3, image2))

    k = cv2.waitKey(1)
    if k == ord('q'):
      break
    elif k == ord('n'):
      if class_num == 999:
        i += 1
      else:
        with open(os.path.join(dataset_path, folder_name, label_file), 'a') as label:
          label.write(folder_name + '_{:05d}.jpg {}\n'.format(i, class_num))
        cv2.imwrite(os.path.join(dataset_path, folder_name, parts[0], folder_name + '_{:05d}.jpg'.format(i)), image1)
        cv2.imwrite(os.path.join(dataset_path, folder_name, parts[1], folder_name + '_{:05d}.jpg'.format(i)), image2)
        num_buffer = []
        class_num = 999
        i += 1
    elif k == ord('b'):
      i = read_i(dataset_path) - 1
      class_num = read_label(dataset_path)
      delete_file(i)
    elif k >= ord('0') and k <= ord('9'):
      num_buffer.append(k-48)
      class_num = 0
      reverse_list = num_buffer[::-1]
      for j in range(len(reverse_list)):
        class_num += 10**j*reverse_list[j]
    elif k == ord('d'):
      num_buffer= []
      class_num = 999
    cv2.putText(image, str(class_num), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
    cv2.imshow('image', image)  
      
if __name__ == '__main__':
  create_folder(dataset_path)
  main(dataset_path)
