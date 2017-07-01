
from __future__ import division
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import os
import scipy.io

import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.8
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    optimizer = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss)
    return optimizer

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def load_data(path):
    mat = scipy.io.loadmat(path)
    return  mat['x_s']

class data_loader(object):
    def __init__(self, path):
        self.path = path
        self.data = load_data(path)

    def get_data(self):
        return self.data


class neurogan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.data = data_loader(args.dataset_dir)
        self.batch_size = args.batch_size
        self.dataset_dir = args.dataset_dir
        self.mlp_hidden_size = 10
        self.discriminator = discriminator
        self.generator = generator
        self.gen = GeneratorDistribution(range=3)
        self.x_dim  = self.data.get_data().shape[1]
        self.z_dim = 5
        self.learning_rate = args.learning_rate
        self.num_examples_test =args.num_examples_test

        self.display_interval = args.display_interval
        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        # In order to make sure that the discriminator is providing useful gradient
        # information to the generator from the start, we're going to pretrain the
        # discriminator using a maximum likelihood objective. We define the network
        # for this pretraining step scoped as D_pre.
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.x_dim))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = self.discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, self.z_dim))
            self.G,self.g_sum, self.layers = self.generator(self.z, self.mlp_hidden_size,self.x_dim)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 18))
            self.D1 = self.discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.mlp_hidden_size)

        # Define the loss for discriminator and generator networks, and create optimizers for both
        #self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_d = 0.5*(tf.reduce_mean((self.D1-1)**2 +tf.reduce_mean(self.D2)**2))

        #self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        self.loss_g = 0.5* tf.reduce_mean((self.D2)**2)
        #tf.summary.scalar('loss_d', self.loss_d)
        #tf.summary.scalar('loss_g', self.loss_g)

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_g)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.loss_d)
        self.G1_sum = tf.summary.histogram('G1', self.G)

        self.g1_sum = tf.summary.merge([self.g_loss_sum,self.G1_sum,self.g_sum ])
        self.d1_sum = tf.summary.merge([self.d_loss_sum])

    def train(self, args):
        """Train """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        for epoch in range(args.epochs):
            # update discriminator
            x = self.data.get_data()
            #plt.plot(np.mean(x,0))
            #plt.show()
            batch_idxs = x.shape[0] // self.batch_size
            for j in range(batch_idxs):
                current_batch_x = x[j * self.batch_size:(j + 1) * self.batch_size, :]
                current_batch_z =  self.gen.sample(self.batch_size)
                #current_batch_z = current_batch_z.reshape((self.batch_size,1))
                current_batch_z = np.random.normal(size=[self.batch_size, self.z_dim]).astype(np.float32)
                #print current_batch_z
                loss_d, _ , summary_str= self.sess.run([self.loss_d, self.opt_d,self.d1_sum], {
                    self.x: current_batch_x,
                    self.z: current_batch_z
                })
                self.writer.add_summary(summary_str, counter)

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_g, _ ,summary_str= self.sess.run([ self.loss_g, self.opt_g, self.g1_sum], {
                    self.z: current_batch_z
                })

                self.writer.add_summary(summary_str, counter)

                counter += 1

            if epoch % self.display_interval == 0:
                print('{}: {}\t{}'.format(epoch, loss_d, loss_g))

                #for k in range(len(g)):
                #    print ('Layer number ', k)
                #    print g[k][0,:]
                g = self.sample(self.num_examples_test, self.sess)

                print (np.mean(g[-1], axis=0))
                #plt.plot(np.mean(g[-1], axis=0))
                #plt.show()
        self.save(args.checkpoint_dir, counter)
        g = self.sample(self.num_examples_test, self.sess)

        plt.plot(g[-1].T)
        plt.show()

    def save(self, checkpoint_dir, step):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample(self, num_examples, sess):
        #zs = np.linspace(-self.gen.range, self.gen.range, num_examples)
        zs = np.random.normal(size=[self.batch_size*num_examples, self.z_dim]).astype(np.float32)

        g = np.zeros((num_examples, self.x_dim))
        for i in range(num_examples // self.batch_size):
            d = sess.run([self.G,self.layers], {
                self.z:
                    zs[self.batch_size * i:self.batch_size * (i + 1),:]
            })
            g[self.batch_size * i:self.batch_size * (i + 1), :] = d[0]

        return  d[1]

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        x = self.data.get_data()

        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        g= self.sample(self.num_examples_test, self.sess)
        for i in range(len(g)):
            print ('------------lAyer number ------------', i)
            print np.mean(g[i][:,:], 0)
        for j in range(g[-2].shape[0]):
            fig, ax = plt.subplots()
            ax.plot(g[-1][j,:], ':o')
            ax.plot(x[j,:], ':o')

            if j>1:
                break
        #for i in range(g.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(np.mean(g[-1][:,:], axis=0),':o')
        ax.plot(np.mean(x,0))
        plt.show()
