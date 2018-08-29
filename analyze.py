from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import math
import os
from scipy import misc
from feature_extractor.feature_extractor import FeatureExtractor
import feature_extractor.utils as utils

video_name = '/home/cheer/video_test/test3.mp4'
path = '/home/cheer/video_test/corre/data'
label_path = '/home/cheer/video_test/corre/data/overlap/label_name.txt'

def nothing(emp):
  pass

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

def make_dirs(path):
  if not os.path.exists(os.path.join(path, 'result', 't0')):
    os.makedirs(os.path.join(path, 'result', 't0'))
  if not os.path.exists(os.path.join(path, 'result', 't1')):
    os.makedirs(os.path.join(path, 'result', 't1'))
  if not os.path.exists(os.path.join(path, 'result', 't2')):
    os.makedirs(os.path.join(path, 'result', 't2'))

def find_max(boxes_nms):
  if len(boxes_nms) == 0:
    return []
  boxes = []
  for box_nms in boxes_nms:
    box_nms = np.append(box_nms, (box_nms[2]-box_nms[0])*(box_nms[3]-box_nms[1]))
    boxes.append(box_nms)
  boxes = np.array(boxes)
  idx = np.argsort(boxes[:,4])
  x_center = boxes[idx[-1]][0] + (boxes[idx[-1]][2] - boxes[idx[-1]][0]) / 2
  y_center = boxes[idx[-1]][1] + (boxes[idx[-1]][3] - boxes[idx[-1]][1]) / 2
  box_max = np.append(boxes[idx[-1]], [x_center, y_center])
  box_max = np.array(box_max, dtype = np.int32)
  return box_max

def non_max_suppression(boxes, overlapThresh):
  if len(boxes) == 0:
    return []
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  pick = []
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / area[idxs[:last]] 
    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

  return boxes[pick].astype("int")

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  print("SSIM: {}".format(score))

  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]

  return diff, thresh, cnts, score

def convert_box(cnts):
  box = []
  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 6 and h > 15: 
      box.append([x, y, x+w, y+h])
  box = np.array(box)
  return box

def find_overlap(box_max, box_o):
  x1 = min(box_max[0], box_o[0])
  y1 = min(box_max[1], box_o[1])
  x2 = max(box_max[2], box_o[2])
  y2 = max(box_max[3], box_o[3])
  box = [x1, y1, x2, y2]
  box = np.array(box, dtype = np.int32)
  print (box)
  return box

def start(video_name):
  box_o = [0,0,0,0]
  cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
  cap = cv2.VideoCapture(video_name)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cv2.createTrackbar('time', 'frame', 0, frames, nothing)
  loop_flag = 0
  pos = 0
  label_name = []

  with open(label_path) as label_file:
    labels = label_file.readlines()
  for label in labels:
    label_name.append(label.strip().split(':')[1])
  print(label_name)

  feature_extractor = FeatureExtractor(
    network_name='flownet_s',
    checkpoint_path='/home/cheer/video_test/corre/flownet_s/model.ckpt-12000',
    batch_size=1,
    num_classes=30,
    preproc_func_name='flownet_s')
  feature_extractor.print_network_summary()

  if cap.isOpened():
    ret, frameA = cap.read()
  while(cap.isOpened()):
    if loop_flag == pos:
      loop_flag = loop_flag + 1
      cv2.setTrackbarPos('time', 'frame', loop_flag)
    else:
      pos = cv2.getTrackbarPos('time', 'frame')
      loop_flag = pos
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frameB = cap.read()
    diff, thresh, cnts, score= compare_frame(frameA, frameB)
    frameC = frameB.copy()
    frameD = frameB.copy()
    boxes = convert_box(cnts)
    boxes_nms = non_max_suppression(boxes, 0.3)
    box_max = find_max(boxes_nms)

    if len(box_max):
      box_size = box_max[4]
      box_center = box_max[5:7]
      overlap = find_overlap(box_max[0:4], box_o)
      cv2.rectangle(frameC, (box_max[0], box_max[1]), (box_max[2], box_max[3]), (0, 0, 255), 2)
      cv2.rectangle(frameC, (overlap[0], overlap[1]), (overlap[2], overlap[3]), (0, 255, 0), 2)
      cv2.imwrite('/home/cheer/video_test/corre/data/result/t0/' + '{:05}'.format(pos) + '.jpg', frameA[overlap[1]:overlap[3], overlap[0]:overlap[2]])
      cv2.imwrite('/home/cheer/video_test/corre/data/result/t1/' + '{:05}'.format(pos) + '.jpg', frameD[overlap[1]:overlap[3], overlap[0]:overlap[2]])
      image1 = '/home/cheer/video_test/corre/data/result/t0/' + '{:05}'.format(pos) + '.jpg'
      image2 = '/home/cheer/video_test/corre/data/result/t1/' + '{:05}'.format(pos) + '.jpg'
      clip_class = classification_placeholder_input(feature_extractor, image1, image2, 'Logits',1, 6)
      cv2.putText(frameC, label_name[clip_class] , (25,232), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
      cv2.imwrite('/home/cheer/video_test/corre/data/result/t2/' + '{:05}'.format(pos) + '.jpg', frameC)
      box_o = box_max[0:4]
    cv2.imshow("frame", frameC)
    frameA = frameB.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  make_dirs(path)
  start(video_name)
