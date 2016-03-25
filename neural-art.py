import argparse
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from vgg import VGG
import neural_config
from vgg19 import VGG19

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
  parser.add_argument('--out_dir', default="output")
  args = parser.parse_args()
  return args.content, args.style, args.modelpath, args.width, args.iters, args.out_dir

def read_image(path, w):
  img = scipy.misc.imread(path).astype(np.float)
  img = scipy.misc.imresize(img, (w, w))
  img = img.astype(np.float32)

  red, green, blue = np.split(img, 3, 2)
  img = np.concatenate([
            red - neural_config.mean[0],
            green - neural_config.mean[1],
            blue - neural_config.mean[2],
        ], 2)
  return img

def save_image(im, step, out_dir):
  img = im.copy()
  red, green, blue = np.split(img, 3, 3)
  img = np.concatenate([
            red + neural_config.mean[0],
            green + neural_config.mean[1],
            blue + neural_config.mean[2],
        ], 3)
  img = np.clip(img[0, ...], 0, 255).astype(np.uint8)
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  scipy.misc.imsave("{}/neural_art_step{}.png".format(out_dir, step), img)

def getContentValues(content_image):
  print "Load content feat map..."
  # image_data = np.reshape(content_image, [1, neural_config.image_size, neural_config.image_size, 3])
  image = tf.constant(content_image)
  image = tf.reshape(image, [1, neural_config.image_size, neural_config.image_size, 3])
  net = VGG19({'data': image}, scope="CONTENT_IMAGE")
  with tf.Session() as sess:
    # Load the data
    net.load(neural_config.model_path, sess)

    # Forward pass
    feature_map = sess.run(net.layers[neural_config.content_layer])

    return feature_map


def getStyleValues(style_image):
  print "Load style feat maps..."

  # image_data = np.reshape(style_image, [1, neural_config.image_size, neural_config.image_size, 3])
  # feed_dict = { "Placeholder:0": image_data }
  image = tf.constant(style_image)
  image = tf.reshape(image, [1, neural_config.image_size, neural_config.image_size, 3])
  net = VGG19({'data': image}, scope="STYLE_IMAGE")
  with tf.Session() as sess:
    # Load the data
    net.load(neural_config.model_path, sess)

    # Forward pass
    feature_maps = sess.run([net.layers[tensor] for tensor in  neural_config.style_layers])

    grams = []

    for index, layer in enumerate(neural_config.style_layers):
      feat_map = feature_maps[index]
      features = np.reshape(feat_map, (-1, feat_map.shape[3]))
      gram = np.matmul(features.T, features) / features.size
      grams.append(gram)

    return grams

def build_graph(content_feat_map, style_grams, content_image):
  print "Make graph for new image..."
  gen_image = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=True, name='gen_image')
    # gen_image = tf.Variable(tf.random_normal([1, neural_config.output_size, neural_config.output_size, 3]) * 0.256, trainable=True, name='gen_image')
  gen_image = tf.reshape(gen_image, [1, neural_config.image_size, neural_config.image_size, 3])
  net = VGG19({'data': gen_image}, scope="GEN_IMAGE")
  with tf.Session() as sess:
    # Load the data
    net.load(neural_config.model_path, sess)

    # Forward pass
    feature_maps = [net.layers[tensor] for tensor in neural_config.new_image_layers]

    img_content_feat_map = feature_maps[0]

    img_style_grams = []

    for index, layer in enumerate(neural_config.style_layers):
      feat_map = feature_maps[index + 1]
      layer_shape = feat_map.get_shape().dims
      size = layer_shape[1].value * layer_shape[1].value * layer_shape[3].value
      features = tf.reshape(feat_map, (-1, layer_shape[3].value))
      gram = tf.matmul(tf.transpose(features), features) / size
      img_style_grams.append(gram)


    # content loss
    content_loss = neural_config.content_weight * tf.nn.l2_loss(
      img_content_feat_map - content_feat_map)

    tf.scalar_summary("content_loss", content_loss)

    # style loss
    style_loss = 0

    for index, style_layer in enumerate(neural_config.style_layers):
      style_loss = tf.nn.l2_loss(
        img_style_grams[index] - style_grams[index]) / 2

    style_loss /= len(neural_config.style_layers)
    style_loss *= neural_config.style_weight


    tf.scalar_summary("style_loss", style_loss)

    # overall loss
    loss = tf.add(content_loss, style_loss)

    tf.scalar_summary("total_loss", loss)

    # optimizer setup
    opt = tf.train.AdamOptimizer(neural_config.learning_rate).minimize(loss)

    init = tf.initialize_all_variables()
    sess.run(init)

    gen_image_value = sess.run(gen_image)
    img_content_feat_map_value = sess.run(img_content_feat_map)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(neural_config.train_dir)

    # optimization
    best_loss = float('inf')
    best = None

    for step in xrange(neural_config.max_iter):
      _, content_loss_value, style_loss_value, loss_value = sess.run([opt, content_loss, style_loss, loss])

      format_str = ('step %d, content_loss_value = %.2f, style_loss_value = %.2f, loss = %.2f')
      print (format_str % (step, content_loss_value, style_loss_value, loss_value))

      last_step = step == neural_config.max_iter - 1
      if (neural_config.checkpoint_steps and step % neural_config.checkpoint_steps == 0) or last_step:
        if loss_value < best_loss:
          best_loss = loss_value
          best = sess.run(gen_image)
          best.reshape([neural_config.image_size, neural_config.image_size, 3])
          save_image(best, step, neural_config.output_dir)

          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)




def main():
  # content_image_path, style_image_path, model_path, output_size, alpha, beta, num_iters, out_dir = parseArgs()
  print "Read images..."
  content_image = read_image(neural_config.content_path, neural_config.image_size)
  style_image   = read_image(neural_config.style_path, neural_config.image_size)

  # vgg = VGG(neural_config.image_size)
  # image = tf.placeholder("float", [1, neural_config.image_size, neural_config.image_size, 3])
  # net = VGG19({'data': image}, scope="FORWARD_IMAGE")
  # with tf.Session() as sess:
  #   Load the data
    # net.load(neural_config.model_path, sess)

  content_feat_map = getContentValues(content_image)
  style_grams = getStyleValues(style_image)

  build_graph(content_feat_map, style_grams, content_image)

  # vgg.printTensors()


if __name__ == '__main__':
    main()