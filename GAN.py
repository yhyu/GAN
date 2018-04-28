import json
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, model_from_json
from keras.layers import InputLayer, Reshape, Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from keras.utils.generic_utils import get_custom_objects

class GAN(object):
    def __init__(self, conf = ""):
        if len(conf) > 0:
            with open(conf, "r") as conf_file:
                conf_data = json.loads(conf_file.read())

                self.encode_size = conf_data["encode_size"]
                self.image_shape = tuple(conf_data["image_shape"])
                self.gen_NN = conf_data["generator"]
                self.dis_NN = conf_data["discriminator"]

        get_custom_objects().update({'log_softmax': Activation(log_softmax)})

    def mkLayer(self, layer):
        strides = (1, 1)
        if 'strides' in layer:
            strides = layer['strides']
            if type(strides) is list:
                strides = tuple(strides)

        activation = None
        if 'activation' in layer:
            activation = layer['activation']

        layer_type = layer["layer"]
        if  layer_type == 'Dense':
            return Dense(int(layer['units']), activation=activation)
        elif layer_type == 'Reshape':
            return Reshape(tuple(layer['shape']))
        elif layer_type == 'Conv2DTranspose':
            return Conv2DTranspose(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'], activation=activation)
        elif layer_type == 'Conv2D':
            return Conv2D(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'], activation=activation)
        elif layer_type == 'MaxPooling2D':
            return MaxPooling2D(pool_size=tuple(layer['size']))
        elif layer_type == 'AveragePooling2D':
            return AveragePooling2D(pool_size=tuple(layer['size']))
        elif layer_type == 'BatchNormalization':
            return BatchNormalization()
        elif layer_type == 'Flatten':
            return Flatten()
        elif layer_type == 'Dropout':
            return Dropout(layer['rate'])

        assert False, "Not supported layer type: %s" % (layer_type)

    def getOptimizer(self, optimizer, learning_rate, loss, var_list):
        if optimizer == "Adam":
            return tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
        if optimizer == "SGD":
            return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)

        assert False, "Not supported optimizer: %s" % (optimizer)

    
    def mkGenerator(self):
        _, _, channels = self.image_shape

        generator = self.generator = Sequential()

        # input data is encode size
        generator.add(InputLayer([self.encode_size]))

        for layer in self.gen_NN["layers"]:
            generator.add(self.mkLayer(layer))

        assert generator.output_shape[1:] == self.image_shape, "generator output (%s) mismatched image shape %s" % (generator.output_shape[1:], self.image_shape)


    def mkDiscriminator(self):
        discriminator = self.discriminator = Sequential()

        # input is images, either real or fake images.
        discriminator.add(InputLayer(self.image_shape))

        for layer in self.dis_NN["layers"]:
            discriminator.add(self.mkLayer(layer))

        discriminator.add(Dense(2))
        discriminator.add(Activation(log_softmax))


    def mkNetwork(self):
        K.clear_session()
        K.set_learning_phase(1)

        input_gen = self.input_gen = tf.placeholder('float32',[None, self.encode_size])

        height, width, channels = self.image_shape
        input_dis = self.input_dis = tf.placeholder('float32', [None, height, width, channels])

        self.mkGenerator()
        self.mkDiscriminator()


    def compile(self, gen_optimizer = None, gen_learning_rate = 0,
                dis_optimizer = None, dis_learning_rate = 0):

        self.mkNetwork()

        gen_opt = self.gen_NN["optimizer"]["name"]
        gen_lr = self.gen_NN["optimizer"]["learning_rate"]
        if gen_optimizer != None:
            gen_opt = gen_optimizer
            gen_lr = gen_learning_rate

        dis_opt = self.dis_NN["optimizer"]["name"]
        dis_lr = self.dis_NN["optimizer"]["learning_rate"]
        if dis_optimizer != None:
            dis_opt = dis_optimizer
            dis_lr = dis_learning_rate

        # D(X), where X is a image
        img_log_prob = self.discriminator(self.input_dis)

        # D(G(Z)), where Z is a distribution
        gen_log_prob = self.discriminator(self.generator(self.input_gen))

        # let say class 0 is fake, class 1 is real...

        # maximize the probability that fake image is "thought" as real
        self.loss_gen = -tf.reduce_mean(gen_log_prob[:,1])

        # maximize the probability that fake image is "thought" as fake but real image as real
        self.loss_dis = -tf.reduce_mean(img_log_prob[:,1] + gen_log_prob[:,0])

        # fix discriminator when train generator
        self.opt_gen = self.getOptimizer(gen_opt, gen_lr,
                                         self.loss_gen, self.generator.trainable_weights)

        # fix generator when train discriminator
        self.opt_dis = self.getOptimizer(dis_opt, dis_lr,
                                         self.loss_dis, self.discriminator.trainable_weights)


    def batch_evaluate(self, sess, batch_images, iter_gen = 1, iter_dis = 1):
        feed_dict = {
            self.input_gen: np.random.normal(size=(batch_images.shape[0], self.encode_size)).astype('float32'),
            self.input_dis: batch_images
        }

        loss_gen = 0.0
        loss_dis = 0.0
        for _ in range(iter_dis):
            loss_dis += sess.run([self.opt_dis, self.loss_dis], feed_dict)[1]
        loss_dis /= iter_dis

        for _ in range(iter_gen):
            loss_gen += sess.run([self.opt_gen, self.loss_gen], feed_dict)[1]
        loss_gen /= iter_gen

        return loss_gen, loss_dis


    def getGenerator(self):
        return self.generator

    def getDiscriminator(self):
        return self.discriminator

    def load_model(self, gen_json, dis_json):
        K.clear_session()
        K.set_learning_phase(0)

        with open(gen_json, "r") as gen_json_file:
            self.generator = model_from_json(gen_json_file.read())

        with open(dis_json, "r") as dis_json_file:
            self.discriminator = model_from_json(dis_json_file.read())


    def load_generator_weights(self, file_name):
        self.generator.load_weights(file_name)

    def load_discriminator_weights(self, file_name):
        self.discriminator.load_weights(file_name)

    def save_generator_model(self, file_name):
        model_json = self.generator.to_json()
        with open(file_name, "w") as json_file:
            json_file.write(model_json)

    def save_discriminator_model(self, file_name):
        model_json = self.discriminator.to_json()
        with open(file_name, "w") as json_file:
            json_file.write(model_json)

    def save_genenerator_weights(self, file_name):
        self.generator.save_weights(file_name)

    def save_discriminator_weights(self, file_name):
        self.discriminator.save_weights(file_name)


def log_softmax(x):
    return tf.nn.log_softmax(x)
    