import cv2
import numpy as np
import os

video_path = '/home/cheer/video_test/videos'
name = '1.mp4'
output_path = '/home/cheer/video_test/corre/frames'

def nothing(emp):
  pass

def make_dirs():
  if not os.path.exists(os.path.join(output_path, name)):
    print 'create path {}'.format(os.path.join(output_path, name))
    os.makedirs(os.path.join(output_path, name))
  else:
    print 'path {} exist'.format(os.path.join(output_path, name))

def start():
  make_dirs()
  video = os.path.join(video_path, name)
  cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
  cap = cv2.VideoCapture(video)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cv2.createTrackbar('time', 'frame', 0, frames, nothing)
  loop_flag = 0
  pos = 0
  while(cap.isOpened()):
    if loop_flag == pos:
      loop_flag = loop_flag + 1
      cv2.setTrackbarPos('time', 'frame', loop_flag)
    else:
      pos = cv2.getTrackbarPos('time', 'frame')
      loop_flag = pos
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frameB = cap.read()
    cv2.imwrite(os.path.join(output_path, name, '{:05}'.format(pos) + '.jpg'), frameB)
    cv2.imshow("frame", frameB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  start()
