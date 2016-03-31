import cv2
import numpy as np
import neural_config
import argparse
import os
import tensorflow as tf
import neural_config


def read_image(path):
  img = cv2.imread(path)
  h, w, c = np.shape(img)
  scale_size = neural_config.scale_size
  crop_size = neural_config.crop_size

  assert c == 3

  aspect = float(w)/h
  if w < h:
    resize_to = (scale_size, int((1.0/aspect)*scale_size))
  else:
    resize_to = (int(aspect*scale_size), scale_size)

  img = cv2.resize(img, resize_to)
  img = img.astype(np.float32)
  for i in range(3):
    img[:,:,i] -= neural_config.mean[i]
  # img -= np.array(neural_config.mean)

  h, w, c = img.shape
  ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
  img = img[ho:ho+crop_size, wo:wo+crop_size, :]
  img = img[None, ...]

  return img

def save_image(im, step, out_dir):
  img = im.copy()
  for i in range(3):
    img[0,:,:,i] += neural_config.mean[i]
  # img += np.array(neural_config.mean)

  img = np.clip(img[0, ...], 0, 255).astype(np.uint8)
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  cv2.imwrite("{}/neural_art_step{}.png".format(out_dir, step), img)

  return img


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--content', '-c', default=neural_config.content_path,
                      help='Content image path')
  parser.add_argument('--style', '-s', default=neural_config.style_path,
                      help='Style image path')
  parser.add_argument('--iters', '-i', default=neural_config.max_iter, type=int,
                      help='Number of steps/iterations')
  parser.add_argument('--output_dir', default=neural_config.output_dir)
  args = parser.parse_args()
  return args.content, args.style, args.iters, args.output_dir


# def read_image(path, w=None):
#   img = scipy.misc.imread(path).astype(np.float)
#   if w:
#     img = scipy.misc.imresize(img, (w, w))
#   img = img.astype(np.float32)
#
#   red, green, blue = np.split(img, 3, 2)
#   img = np.concatenate([
#             red - neural_config.mean[0],
#             green - neural_config.mean[1],
#             blue - neural_config.mean[2],
#         ], 2)
#   return img

def activation_summary(x, tensor_name):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

