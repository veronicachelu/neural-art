import argparse
import numpy as np
import os
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_size', 224,
                           """the size of the image that goes into the VGG net""")
def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--modelpath', '-mp', default='vgg',
                      help='Model file path')
  parser.add_argument('--content', '-c', default='images/content.jpg',
                      help='Content image path')
  parser.add_argument('--style', '-s', default='images/style.jpg',
                      help='Style image path')
  parser.add_argument('--width', '-w', default=800, type=int,
                      help='Output image width')
  parser.add_argument('--iters', '-i', default=5000, type=int,
                      help='Number of iterations')
  parser.add_argument('--alpha', '-a', default=1.0, type=float,
                      help='alpha (content weight)')
  parser.add_argument('--beta', '-b', default=200.0, type=float,
                      help='beta (style weight)')
  parser.add_argument('--out_dir', default="output")
  args = parser.parse_args()
  return args.content, args.style, args.modelpath, args.width, args.alpha, args.beta, args.iters, args.out_dir


def read_image(path, w=None):
  img = scipy.misc.imread(path).astype(np.float)
  # Resize if ratio is specified
  if w:
    r = w / np.float32(img.shape[1])
    img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
  img = img.astype(np.float32)
  img = img[None, ...]
  return img


def main():
  print "Read images..."



if __name__ == '__main__':
    main()