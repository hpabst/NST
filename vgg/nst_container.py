from vgg.vgg import *
import tensorflow as tf
import numpy as np
import scipy
from scipy.misc import imread, imresize
import os
import sys


class NST(object):
    """
    Container around neural style-transfer operations.
    """

    def __init__(self, content_path=None, style_path=None, content_ratio=0.6, logdir="tblog/",
                 outputdir="output/", cost_alpha = 60, cost_beta = 40):
        """
        :param content_path: Full filepath to content image.
        :param style_path: Full filepath to style image.
        :param content_ratio: Content seeding ratio for initializtion of generated image (range of 0.0-1.0).
        :param logdir: Tensorboard log directory for debugging.
        :param outputdir: Generated image output directory for debugging.
        :param cost_alpha: Content cost scaling hyperparameter.
        :param cost_beta: Style cost scaling hyperparameter.
        """
        self.CONTENT_PATH = content_path
        self.STYLE_PATH = style_path
        self.VGG16_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        self.CONTENT_RATIO = content_ratio
        self.LOGDIR = logdir
        self.OUTPUTDIR = outputdir
        self.EPOCH_COMPLETE = 0
        self.MAX_EPOCH = 1500
        self.PAUSE = False
        self.LATEST_GEN = None
        self.COST_ALPHA = cost_alpha
        self.COST_BETA = cost_beta
        return

    def neural_style_transfer(self, content_path=None, style_path=None, cost_alpha=None, cost_beta=None, content_ratio=None):
        """
        Compelte neural style transfer between provided content image and style image.
        :param content_path: Full filepath to content image.
        :param style_path: Full filepath to style image.
        :return: Latest trained generated image.
        """
        if content_path is None:
            content_path = self.CONTENT_PATH
        if style_path is None:
            style_path = self.STYLE_PATH
        if cost_alpha is None:
            cost_alpha = self.COST_ALPHA
        if cost_beta is None:
            cost_beta = self.COST_BETA
        if content_ratio is None:
            content_ratio = self.CONTENT_RATIO
        self.EPOCH_COMPLETE = 0
        self.PAUSE = False
        self.LATEST_GEN = None
        tf.reset_default_graph()
        session = tf.Session()
        content_image = self.reshape_and_normalize_image(imread(content_path, mode="RGB"))
        # Reshape the style image into the same dimensions as the content image so we don't need 2 VGG16 models
        # when calculating the cost.
        style_image = self.reshape_and_normalize_image(
            imresize(imread(style_path, mode="RGB"), size=content_image.shape[1:]))
        generated_image = self.generated_image_base(content_image, content_ratio)
        content_model = vgg_model(content_image.shape, weights_path="vgg/vgg16_weights.npz")
        session.run(content_model.get('input').assign(content_image))
        out = content_model.get('conv4_2')
        intermediate_content = session.run(out)  # Intermediate VGG16 output of content image.
        intermediate_generated = out  # Intermediate VGG16 output of generated image.
        cont_cost = self.content_cost(intermediate_content, intermediate_generated)
        session.run(content_model.get('input').assign(style_image))
        style_cost = self.style_cost(content_model, session)
        tot_cost = self.total_cost(cont_cost, style_cost, alpha = cost_alpha, beta=cost_beta)
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(tot_cost)
        session.run(tf.global_variables_initializer())
        session.run(content_model.get("input").assign(generated_image))
        print("Starting image training.")
        for i in range(self.MAX_EPOCH):
            if self.PAUSE:
                break
            session.run(train_step)
            generated_image = session.run(content_model.get("input"))
            curr_total, curr_content, curr_style = session.run([tot_cost, cont_cost, style_cost])
            if i % 20 == 0:
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(curr_total))
                print("content cost = " + str(curr_content))
                print("style cost = " + str(curr_style))
                self.LATEST_GEN = generated_image + self.VGG16_MEANS
            print("Epoch " + str(i) + " complete")
            self.EPOCH_COMPLETE += 1
        return self.LATEST_GEN

    def gram_matrix(self, a):
        """
        Calculates gram matrix of provided matrix.
        :param a: Matrix to calculate gram matrix of.
        :return: Gram matrix of A.
        """
        return tf.matmul(a, tf.transpose(a))

    def generated_image_base(self, content_image, content_ratio):
        """
        Initialize generated image with white noise and seed with content image.
        :param content_image: Content image.
        :return: Initialized generated image.
        """
        noise_image = np.random.uniform(-20, 20, content_image.shape).astype('float32')
        noise_image = noise_image * (1-content_ratio) + content_image * content_ratio
        return noise_image

    def content_cost(self, act_content, act_gener):
        """
        Determines the content cost of the current model.
        :param act_content: Tensor of shape [1, height, width, num channels] representing content image activations.
        :param act_gener: Tensor of shape [1, height, width, num channels] representing generated image activations.
        :return: content cost of current generated image.
        """
        m, height, width, channels = act_gener.get_shape().as_list()
        AcUnroll = tf.reshape(act_content, [channels, height * width])
        AgUnroll = tf.reshape(act_gener, [channels, height * width])
        content = tf.reduce_sum(tf.square(tf.subtract(AcUnroll, AgUnroll)))
        content *= (1 / (4 * height * width * channels))
        return content

    def single_layer_style_cost(self, As, Ag):
        """
        Determines the style cost of the current generated image for the given activations.
        :param As: Tensor of shape [1, height, width, num_channels] representing style image activations.
        :param Ag: Tensor of shape [1, height, width, num_channels] representing generated image activations.
        :return: style cost of current generated image.
        """
        m, nH, nW, nC = Ag.get_shape().as_list()
        As = tf.transpose(tf.reshape(As, [nH * nW, nC]))
        Ag = tf.transpose(tf.reshape(Ag, [nH * nW, nC]))
        gramS = self.gram_matrix(As)
        gramG = self.gram_matrix(Ag)
        style = tf.reduce_sum(tf.square(tf.subtract(gramS, gramG)))
        scalar = (1 / (4 * (nC * nC) * ((nH * nW) * (nH * nW))))
        style = style * scalar
        return style

    def style_cost(self, model, session):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # initialize the overall style cost
        J_style = 0
        STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        # Intermediate layers and weighting taken from original paper. Can make changes if you want.
        for layer_name, coeff in STYLE_LAYERS:
            out = model[layer_name]
            a_S = session.run(out)
            a_G = out
            J_style_layer = self.single_layer_style_cost(a_S, a_G)
            J_style += coeff * J_style_layer

        return J_style

    def total_cost(self, content_cost, style_cost, alpha=0.6, beta=0.4):
        """
        Calculate total cost based on separate content and style costs.
        :param content_cost:
        :param style_cost:
        :param alpha: Content cost scaling hyperparameter.
        :param beta: Style cost scaling hyperparameter
        :return: Total cost
        """
        return alpha * content_cost + beta * style_cost

    def reshape_and_normalize_image(self, image):
        """
        Reshapes and normalizes image to match expected input of VGG16.
        """
        image = np.reshape(image, ((1,) + image.shape))
        return image - self.VGG16_MEANS

    def save_image(self, path, image):
        """
        Save the given image in the provided path.
        :param path: Path to save image in.
        :param image: Unnormalized image.
        :return:
        """
        image = image + self.VGG16_MEANS
        image = np.clip(image[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, image)
        return
