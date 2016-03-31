# scale_size = 256
content_path = "./images/content.jpg"
style_path = "./images/style.jpg"
# alpha = 1.0
# beta = 200.0
max_iter = 1000
# output_size = 800
output_dir = "./output"
content_layer = "conv4_2"
style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
new_image_layers = ["conv4_2", "conv1_1", "conv2_1", "conv3_1",
                    "conv4_1", "conv5_1"]
content_weight = 5e0
style_weight = 1e2
learning_rate = 1e1
decay_steps = 200
decay_rate = 0.94
checkpoint_steps = 100
model_path = './models/vgg19.npy'
train_dir = './train'
mean = [123.68,116.779, 103.939]
# VGG 19 params
batch_size = 25
scale_size = 256
crop_size = 224
isotropic = True