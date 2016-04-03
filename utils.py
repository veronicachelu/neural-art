import cv2
import numpy as np
import neural_config
import argparse
import os
import tensorflow as tf
import neural_config

def add_mean(img):
  for i in range(3):
    img[0,:,:,i] += neural_config.mean[i]

def sub_mean(img):
  for i in range(3):
    img[:,:,i] -= neural_config.mean[i]

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
  sub_mean(img)

  # img -= np.array(neural_config.mean)

  h, w, c = img.shape
  ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
  img = img[ho:ho+crop_size, wo:wo+crop_size, :]
  img = img[None, ...]

  return img

def save_image(im, step, out_dir, output_image_name):
  img = im.copy()
#   add_mean(img)

  img = np.clip(img[0, ...], 0, 255).astype(np.uint8)
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  cv2.imwrite("{}/{}_step_{}.png".format(out_dir, output_image_name, step), img)

  return img


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--content', '-c', default=neural_config.content_path,
                      help='Content image path')
  parser.add_argument('--style', '-s', default=neural_config.style_path,
                      help='Style image path')
  parser.add_argument('--iters', '-i', default=neural_config.max_iter, type=int,
                      help='Number of steps/iterations')
  parser.add_argument('--output_dir', '-o', default=neural_config.output_dir)
  parser.add_argument('--content_weight', '-cw', default=neural_config.content_weight, type=float,
                        help='content weight')
  parser.add_argument('--style_weight', '-sw', default=neural_config.style_weight, type=float,
                      help='style weight')
  parser.add_argument('--tv_weight', '-tvw', default=neural_config.tv_weight, type=float,
                    help='tv weight')
  parser.add_argument('--output_image', '-n', default=neural_config.output_image_name,
                    help='output image name')
  args = parser.parse_args()
  return args.content, args.style, args.iters, args.output_dir,\
         args.content_weight, args.style_weight, args.tv_weight, args.output_image


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


def getTensorSize(x):
  return tf.cast(tf.reduce_prod(x.get_shape()), dtype=tf.float32)

