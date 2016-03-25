# # Import the converted model's class
# from vgg19 import VGG19
#
# # Create an instance, passing in the input data
# net = VGG19({'data':my_input_data})
#
# with tf.Session() as sesh:
#     # Load the data
#     net.load('mynet.npy', sesh)
#     # Forward pass
#     output = sesh.run(net.get_output(), ...)