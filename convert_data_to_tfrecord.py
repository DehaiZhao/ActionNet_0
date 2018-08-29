import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

_NUM_VALIDATION = 0

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 4

dataset_dir = '/home/cheer/video_test/corre/merge/jp_d_all'
#parts = ['T0', 'T1']
#parts = ['T2', 'T3']
parts = ['T4', 'T5']
label_name = 'label.txt'


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  directories = []
  class_names = []
  with open(os.path.join(dataset_dir, label_name)) as list_file:
    file_list = list_file.readlines()
  return file_list


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'clip_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  img_files_a = []
  img_files_b = []
  class_names = []

  for filename in filenames:
    img_files_a.append(os.path.join(dataset_dir, parts[0], filename.strip().split()[0]))
    img_files_b.append(os.path.join(dataset_dir, parts[1], filename.strip().split()[0]))
    class_names.append(filename.strip().split()[1])

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_a = tf.gfile.FastGFile(img_files_a[i], 'rb').read()
            image_b = tf.gfile.FastGFile(img_files_b[i], 'rb').read()

            class_id = int(class_names[i])

            example = dataset_utils.image_to_tfexample(
                image_a, image_b, b'jpg', class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def main(_):

  file_list = _get_filenames_and_classes(dataset_dir)

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(file_list)
  training_filenames = file_list[_NUM_VALIDATION:]
  #validation_filenames = file_list[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, dataset_dir)
  #_convert_dataset('validation', validation_filenames, dataset_dir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')

if __name__ == '__main__':
  tf.app.run()
