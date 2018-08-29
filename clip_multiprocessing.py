from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import math
import os
import time
import multiprocessing
from scipy import misc

video_path = '/home/cheer/video_test/videos'
dataset_path = '/home/cheer/video_test/corre/diff_data'
parts = ['t0', 't1']

def nothing(emp):
  pass

def make_dirs(video_name):
  folder_name = os.path.splitext(video_name)[0]
  for part in parts:
    if not os.path.exists(os.path.join(dataset_path, folder_name, part)):
      print 'create folder {}'.format(folder_name)
      os.makedirs(os.path.join(dataset_path, folder_name, part))
    else:
      print 'folder {} exist'.format(folder_name)

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
  box_width = x2 - x1
  box_height = y2 - y1
  box_max_width = box_max[2] - box_max[0]
  box_max_height = box_max[3] - box_max[1]
  box_o_width = box_o[2] - box_o[0]
  box_o_height = box_o[3] - box_o[1]
  box = [x1, y1, x2, y2]
  if box_width > box_max_width*20 and box_width > box_o_width*20:
    box = box_max[0:4]
  if box_height > box_max_height*20 and box_height > box_o_height*20:
    box = box_max[0:4]
  box = np.array(box, dtype = np.int32)
  return box

def start():
  video_name = q.get()
  make_dirs(video_name)
  print 'processing video', video_name
  video = os.path.join(video_path, video_name)
  folder_name = os.path.splitext(video_name)[0]
  box_o = [0,0,0,0]
  cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
  cap = cv2.VideoCapture(video)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cv2.createTrackbar('time', video_name, 0, frames, nothing)
  loop_flag = 0
  pos = 0
  if cap.isOpened():
    ret, frameA = cap.read()
  while(cap.isOpened()):
    if loop_flag == pos:
      loop_flag = loop_flag + 1
      cv2.setTrackbarPos('time', video_name, loop_flag)
    else:
      pos = cv2.getTrackbarPos('time', video_name)
      loop_flag = pos
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frameB = cap.read()
    try:
      diff, thresh, cnts, score= compare_frame(frameA, frameB)
    except:
      break
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
      cv2.imwrite(os.path.join(dataset_path, folder_name, parts[0], '{:05}'.format(pos) + '.jpg'), frameA[box_max[1]:box_max[3], box_max[0]:box_max[2]])
      cv2.imwrite(os.path.join(dataset_path, folder_name, parts[1], '{:05}'.format(pos) + '.jpg'), frameD[box_max[1]:box_max[3], box_max[0]:box_max[2]])
      box_o = box_max[0:4]
    cv2.imshow(video_name, frameC)
    frameA = frameB.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyWindow(video_name)

def test():
  print q.get()
  print q.qsize()

if __name__ == '__main__':
  video_list = os.listdir(video_path)
  q = multiprocessing.Queue()
  pool = multiprocessing.Pool(processes = 8)
  #processes_number = 5
  for video_name in video_list:
    q.put(video_name)
  while q.qsize():
    pool.apply_async(start, ())
  #while q.qsize():
  #  processes = []
  #  for _ in range(processes_number):
  #    p = multiprocessing.Process(target = start, args = (q,))
  #    processes.append(p)
  #    p.start()
  #  for p in processes:
  #    p.join()
  cv2.destroyAllWindows()
