import numpy as np
import tensorflow as tf
from vgg import VGG
import neural_config
from vgg19 import VGG19
import utils


def getContentValues(content_image, scope):
  print "Load content feat map..."
  image = tf.constant(content_image)
  net = VGG19({'data': image}, scope=scope, trainable=False)
  with tf.Session() as sess:
    # Load the data
    net.load(neural_config.model_path, sess)

    # Forward pass
    feature_map = sess.run(net.layers[neural_config.content_layer])

    return feature_map


def getStyleValues(style_image, scope):
  print "Load style feat maps..."

  image = tf.constant(style_image)
  net = VGG19({'data': image}, scope=scope, trainable=False)
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

def build_graph(content_feat_map, style_grams, content_image, scope):
  print "Make graph for new image..."
  initial = tf.random_normal(content_image.shape) * 0.256
  image = tf.Variable(initial, trainable=True)

  image_vars_set = set(tf.all_variables())

  net = VGG19({'data': image}, scope=scope, trainable=False)
  with tf.Session() as sess:
    # Load the data
    net.load(neural_config.model_path, sess)

    # Forward pass
    feature_maps = [net.layers[tensor] for tensor in neural_config.new_image_layers]

    img_content_feat_map = feature_maps[0]
    utils.activation_summary(img_content_feat_map)

    img_style_grams = []

    for index, layer in enumerate(neural_config.style_layers):
      feat_map = feature_maps[index + 1]
      layer_shape = feat_map.get_shape().dims
      size = layer_shape[1].value * layer_shape[1].value * layer_shape[3].value
      features = tf.reshape(feat_map, (-1, layer_shape[3].value))
      gram = tf.matmul(tf.transpose(features), features) / size
      img_style_grams.append(gram)
      utils.activation_summary(gram)

    # content loss
    content_loss = neural_config.content_weight * (2 * tf.nn.l2_loss(
      img_content_feat_map - content_feat_map) / content_feat_map.size)

    tf.scalar_summary("content_loss", content_loss)

    # style loss
    style_loss = 0
    style_losses = []
    for index, style_layer in enumerate(neural_config.style_layers):
      style_losses.append(2 * tf.nn.l2_loss( img_style_grams[index] - style_grams[index]) / style_grams[index].size)

    style_loss += neural_config.style_weight * reduce(tf.add, style_losses)


    tf.scalar_summary("style_loss", style_loss)

    # overall loss
    loss = tf.add(content_loss, style_loss)

    tf.scalar_summary("total_loss", loss)

    temp = set(tf.all_variables())

    # optimizer setup
    # opt = tf.train.AdamOptimizer(neural_config.learning_rate).minimize(loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=neural_config.learning_rate, global_step=global_step,
                                               decay_steps=neural_config.decay_steps,
                                               decay_rate=neural_config.decay_rate,
                                               staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    vars = set(tf.all_variables()) - temp
    vars = vars.union(image_vars_set)
    # vars.add(image)
    init = tf.initialize_variables(vars)

    sess.run(init)

    # optimization
    best_loss = float('inf')

    for step in xrange(neural_config.max_iter):
      _, content_loss_value, style_loss_value, loss_value = sess.run([opt, content_loss, style_loss, loss])

      format_str = ('step %d, content_loss_value = %.2f, style_loss_value = %.2f, loss = %.2f')
      print (format_str % (step, content_loss_value, style_loss_value, loss_value))

      last_step = step == neural_config.max_iter - 1
      if (neural_config.checkpoint_steps and step % neural_config.checkpoint_steps == 0) or last_step:
        if loss_value < best_loss:
          best_loss = loss_value
          best = sess.run(image)

          # best_image = utils.save_image(best, step, neural_config.output_dir)
          tf.image_summary(("images/best%d" % (step)), best)

          summary_op = tf.merge_all_summaries()
          summary_writer = tf.train.SummaryWriter(neural_config.train_dir)

          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)




def main():
  # content_image_path, style_image_path, model_path, output_size, alpha, beta, num_iters, out_dir = parseArgs()
  # clear previous output folders
  if tf.gfile.Exists(neural_config.output_dir):
    tf.gfile.DeleteRecursively(neural_config.output_dir)
  tf.gfile.MakeDirs(neural_config.output_dir)

  if tf.gfile.Exists(neural_config.train_dir):
    tf.gfile.DeleteRecursively(neural_config.train_dir)
  tf.gfile.MakeDirs(neural_config.train_dir)

  print "Read images..."
  content_image = utils.read_image(neural_config.content_path)
  style_image   = utils.read_image(neural_config.style_path)

  content_feat_map = getContentValues(content_image, "Content1")

  style_grams = getStyleValues(style_image, "Style")

  build_graph(content_feat_map, style_grams, content_image, "Gen")


if __name__ == '__main__':
    main()