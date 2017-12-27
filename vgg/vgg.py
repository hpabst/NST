import tensorflow as tf
import numpy as np


def vgg_model(input_size, weights_path="vgg16_weights.npz"):
    """
    Initializes VGG16 CNN architecture and trained weights.
    :param input_size: Expected size of input images.
    :param weights_path: Filepath to npz weights file for VGG16.
    :return:
    """
    print("Initiating vgg16 network.")
    network = {"input": tf.Variable(np.zeros(input_size), dtype=tf.float32)}
    weights = np.load(weights_path)
    vgg_layers = ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2", "conv3_1", "conv3_2",
                  "pool3_3", "pool3", "conv4_1", "conv4_2", "conv4_3", "pool4", "conv5_1", "conv5_2",
                  "conv5_3", "pool5"]
    # NST doesn't require any intermediate layers past the conv layers, so we don't bother putting together
    # the FC and softmax layers of the network.
    current = network["input"]
    for l in vgg_layers:
        print("\tInitiating layer {0}.".format(l))
        if "conv" in l:
            with tf.name_scope(l) as scope:
                w = weights["{0}_W".format(l)]
                b = weights["{0}_b".format(l)]
                w = tf.constant(w)
                b = tf.constant(b)
                conv = tf.nn.conv2d(current, w, [1, 1, 1, 1], padding="SAME")
                conv_plus_bias = tf.nn.bias_add(conv, b)
                conv_relu = tf.nn.relu(conv_plus_bias, name=scope)
                network[l] = conv_relu
                current = conv_relu
        elif "pool" in l:
            with tf.name_scope(l) as scope:
                pool = tf.nn.avg_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      name=scope)
                network[l] = pool
                current = pool
    print("Vgg16 network initialized.")
    return network
