import argparse
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from vgg import VGG
import neural_config

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


def read_image(path, w):
  img = scipy.misc.imread(path).astype(np.float)
  img = scipy.misc.imresize(img, (w, w))
  img = img.astype(np.float32)
  return img


def main():
  # content_image_path, style_image_path, model_path, output_size, alpha, beta, num_iters, out_dir = parseArgs()
  print "Read images..."
  content_image = read_image(neural_config.content_path, neural_config.image_size)
  style_image   = read_image(neural_config.style_path, neural_config.image_size)

  vgg = VGG(neural_config.image_size)

  # content_feat_map = vgg.getContentValues(content_image, neural_config.content_layer)
  # style_feat_maps = vgg.getContentValues(style_image, neural_config.style_layers)
  step, image = vgg.makeImage(content_image, style_image)
  print step
  print image.shape
  # vgg.printTensors()



if __name__ == '__main__':
    main()