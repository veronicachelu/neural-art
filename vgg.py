import tensorflow as tf
import cv2
import numpy as np
import neural_config

class VGG(object):
  def __init__(self, image_size):
    # self.nr_feat_maps = nr_feat_maps
    # self.tensor_names = tensor_names
    self.image_size = image_size
    self.create_graph()

  def create_graph(self):
    with open(neural_config.model_path, mode='rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      images = tf.placeholder("float", [None, self.image_size, self.image_size, 3])
      self.new_image_tensors  = tf.import_graph_def(graph_def, input_map={ "images": images }, return_elements=neural_config.new_image_layers)
      print "graph loaded from disk"

  def printTensors(self):
    graph = tf.get_default_graph()
    operations = graph.get_operations()
    for operation in operations:
      print ("Operation:", operation.name)
      for k in operation.inputs:
          print (operation.name, "Input ", k.name, k.get_shape())
      for k in operation.outputs:
          print (operation.name, "Output ", k.name)
      print ("\n")

  def getContentValues(self, content_image, tensor_name):
    print "Load content feat map..."
    with tf.Session() as sess:
      init = tf.initialize_all_variables()
      images = tf.placeholder("float", [None, self.image_size, self.image_size, 3], name="Placeholder")
      image_data = tf.convert_to_tensor(content_image, dtype=tf.float32).eval()
      image_data = np.reshape(image_data, [1, self.image_size, self.image_size, 3])
      assert image_data.shape == (1, self.image_size, self.image_size, 3)
      feed_dict = { "Placeholder:0": image_data }

      feature_map = sess.run(tensor_name, feed_dict=feed_dict)

      return feature_map

  def getStyleValues(self, style_image, tensor_list):
    print "Load style feat maps..."
    with tf.Session() as sess:
      init = tf.initialize_all_variables()
      images = tf.placeholder("float", [None, self.image_size, self.image_size, 3], name="Placeholder")
      image_data = tf.convert_to_tensor(style_image, dtype=tf.float32).eval()
      image_data = np.reshape(image_data, [1, self.image_size, self.image_size, 3])
      assert image_data.shape == (1, self.image_size, self.image_size, 3)
      feed_dict = { "Placeholder:0": image_data }

      feature_maps = sess.run(tensor_list, feed_dict=feed_dict)
      grams = []

      for index, layer in enumerate(neural_config.style_layers):
        feat_map = feature_maps[index]
        features = np.reshape(feat_map, (-1, feat_map.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        grams.append(gram)

      return grams

  def makeImage(self, content_image, style_image):
    print "Make graph for new image..."

    initial = np.random.normal(0, 1, (1, neural_config.output_size, neural_config.output_size, 3)) * 0.256
    # image_data = tf.Variable(initial)
    image_data = np.reshape(initial, [1, self.image_size, self.image_size, 3])
    assert image_data.shape == (1, self.image_size, self.image_size, 3)
    feed_dict = { "Placeholder:0": image_data }

    # feature_maps = sess.run(neural_config.new_image_layers, feed_dict=feed_dict)

    content_feat_map = self.getContentValues(content_image, neural_config.content_layer)
    style_grams = self.getStyleValues(style_image, neural_config.style_layers)

    # content loss
    content_loss = neural_config.content_weight * (2 * tf.nn.l2_loss(
              self.new_image_tensors[0] - content_feat_map) /
              content_feat_map.size)

    # style loss
    style_loss = 0

    for index, style_layer in enumerate(neural_config.new_image_layers):
      if index == 0:
        continue
      layer = self.new_image_tensors[index]
      layer_shape = layer.get_shape().dims
      feats = tf.reshape(layer, (-1, layer_shape[3].value))
      layer_size = layer_shape[1].value * layer_shape[2].value * layer_shape[3].value
      gram = tf.matmul(tf.transpose(feats), feats) / layer_size
      style_loss = neural_config.style_weight * (2 * tf.nn.l2_loss(gram - style_grams[index - 1]) / style_grams[index - 1].size)

    # overall loss
    loss = content_loss + style_loss

    with tf.Session() as sess:
      # optimizer setup
      vars = [op.outputs[0] for op in tf.get_default_graph().get_operations() if op.outputs and op.outputs[0] and op.outputs[0].name == "add:0"]
      opt = tf.train.AdamOptimizer(neural_config.learning_rate).minimize(vars[0])
      init = tf.initialize_all_variables()
      sess.run(init)

      # optimization
      best_loss = float('inf')
      best = None

      for step in xrange(neural_config.max_iter):
        _, content_loss_value, style_loss_value, loss_value = sess.run([opt, content_loss, style_loss, loss], feed_dict=feed_dict)

        format_str = ('%s: step %d, content_loss_value = %.2f, style_loss_value = %.2f, loss = %.2f')
        print (format_str % (step, content_loss_value, style_loss_value, loss_value))

        last_step = step == neural_config.max_iter - 1
        if (neural_config.checkpoint_steps and step % neural_config.checkpoint_steps == 0) or last_step:
          if loss_value < best_loss:
            best_loss = loss_value
            best = sess.run(image_data)
          yield (
            (None if last_step else step),
            best.reshape([self.image_size, self.image_size, 3])
          )





  # def get_features(self, frame_list):
  #   video_feat_maps = []
  #   with tf.Session() as sess:
  #     init = tf.initialize_all_variables()
  #     sess.run(init)
  #     # print "variables initialized"
  #
  #     images = tf.placeholder("float", [None, self.image_size, self.image_size, 3], name="Placeholder")
  #
  #     preprocessed_frames = []
  #     for frame in frame_list:
  #       image_data = tf.convert_to_tensor(frame, dtype=tf.float32).eval()
  #       image_data = np.reshape(image_data, [1, self.image_size, self.image_size, 3])
  #       assert image_data.shape == (1, self.image_size, self.image_size, 3)
  #       preprocessed_frames.append(image_data)
  #
  #     batch_frames = np.row_stack(preprocessed_frames)
  #     feed_dict = { "Placeholder:0": batch_frames }
  #
  #     tensor_list = []
  #
  #     for tensor in self.tensor_names:
  #       tensor_list.append(sess.graph.get_tensor_by_name(tensor))
  #
  #     video_feat_maps = sess.run(tensor_list, feed_dict=feed_dict)
  #
  #     # video_feat_maps.append(feat_map_list)
  #
  #   return video_feat_maps
