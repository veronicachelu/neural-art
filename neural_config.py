CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0

image_size = 224
#Path to the model needed to create the graph
model_path =  "./models/vgg16.tfmodel"
content_path = "./images/content.jpg"
style_path = "./images/style.jpg"
alpha = 1.0
beta = 200.0
max_iter = 1000
output_size = 224
output_dir = "./output"
content_layer = "import/conv4_2/Relu:0"
style_layers = ["import/conv1_1/Relu:0", "import/conv2_1/Relu:0", "import/conv3_1/Relu:0", "import/conv4_1/Relu:0", "import/conv5_1/Relu:0"]
new_image_layers = ["conv4_2/Relu:0", "conv1_1/Relu:0", "conv2_1/Relu:0", "conv3_1/Relu:0",
                    "conv4_1/Relu:0", "conv5_1/Relu:0"]
content_weight = 5e0
style_weight = 1e2
learning_rate = 1e1
checkpoint_steps = 10