from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(1)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        w_sum = variable_summaries(w)

        b = tf.get_variable('b', [output_dim], initializer=const)
        b_sum = variable_summaries(b)

        return tf.matmul(input, w) + b, w_sum, b_sum

def variable_summaries(var, name ='summaries'):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    mean_sum = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    std_sum = tf.summary.scalar('stddev', stddev)
    max_sum = tf.summary.scalar('max', tf.reduce_max(var))
    min_sum = tf.summary.scalar('min', tf.reduce_min(var))
    hist_sum = tf.summary.histogram('histogram', var)
    total_sum = tf.summary.merge([std_sum, mean_sum,max_sum,min_sum ,hist_sum])
    return  total_sum


def lrelu(x, leak=0.00001, name="lrelu"):
  return tf.maximum(x, leak*x)


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name,center=True)

def conv1d(x,input,filt_shape = (3, 1, 1)):
    h2 = tf.reshape(x, [input.get_shape()[0].value, -1, filt_shape[1]])
    # filter = tf.ones([2,1,1])  # these shold be real values, not 0
    filter = tf.Variable(tf.truncated_normal(shape=filt_shape, mean=1, stddev=0.1))
    output = tf.nn.conv1d(h2, filter, stride=1, padding="VALID")
    fc = tf.reshape(output, [input.get_shape()[0].value, -1])
    return fc

def generator(input, h_dim , x_dim):
    layers = []
    h0, w_sum, b_sum = linear(input, h_dim, 'h0')
    h0 = lrelu(batch_norm(h0,name = 'h0'))
    h0_sum = variable_summaries(h0,name ='h0')

    fc = lrelu(batch_norm(conv1d(h0, input,filt_shape=(3,1,8)),name = 'fc'))
    fc_sum = variable_summaries(fc,name ='conv')

    fc1 = batch_norm(conv1d(fc, input, filt_shape=(3,8,8)))
    fc_sum1 = variable_summaries(fc1, name='conv')
    d1 = deconv1d(tf.nn.relu(fc1), input, stride=1,  output_shape = 8)
    d1 = tf.nn.dropout(batch_norm(d1, 'g_bn_d1'), 0.5)
    d2 = deconv1d(tf.nn.relu(d1), input, stride=1, output_shape=8)
    d2 = tf.nn.dropout(batch_norm(d2, 'g_bn_d2'), 0.5)

    h1, w1_sum , b1_sum = linear(d2, x_dim, 'h1')
    h1 = lrelu(batch_norm(h1, name = 'h1'))
    h1_sum = variable_summaries(h1, name ='h1')
    h2, w2_sum, b2_sum = linear(h1, x_dim, 'h2')
    h2 = lrelu(h2)
    h2_sum = variable_summaries(h2, name='h2')

    h3, w00_sum, b00_sum = linear(h2, x_dim, 'h3')
    h3 = lrelu(h3)

    total_sum = tf.summary.merge([h0_sum, fc_sum, h1_sum,h2_sum])
    layers.append(h0)
    layers.append(fc)
    layers.append(fc1)
    layers.append(d2)
    layers.append(h1)
    layers.append(h2)
    layers.append(h3)

    #layers.append(h3)
    return h3,total_sum,layers

def mlp(input, output_dim , x_dim):
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    return fc3, [w1,b1,w2,b2,w3,b3]

def discriminator(input, h_dim):
    fc = batch_norm(lrelu(conv1d(input, input)),name = 'fc')
    fc1 = batch_norm(lrelu(conv1d(fc, input)),name = 'fc1')

    h0, w_sum, b_sum  =linear(fc1, h_dim * 2, 'd0')
    h0 = lrelu(h0)
    h11 = lrelu(batch_norm(conv1d(h0, input)))
    h12 = lrelu(batch_norm(conv1d(h11, input), name='h12'))

    h1, w_sum, b_sum = linear(h12, h_dim * 2, 'd1')
    h1 = lrelu(h1)
    h2, w_sum, b_sum =linear(h1, h_dim * 2, scope='d2')
    h2 = lrelu(h2)
    h3, w_sum, b_sum = linear(h2, 1, scope='d3')
    h3 = lrelu(h3)
    h3 = tf.sigmoid(h3)
    return h3



def discriminator_old(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

def generator_unet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, options.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = batch_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = batch_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = batch_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = batch_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = batch_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = batch_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = batch_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.concat([tf.nn.dropout(batch_norm(d1, 'g_bn_d1'), 0.5), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.concat([tf.nn.dropout(batch_norm(d2, 'g_bn_d2'), 0.5), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.concat([tf.nn.dropout(batch_norm(d3, 'g_bn_d3'), 0.5), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([batch_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([batch_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([batch_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([batch_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        s = options.image_size
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(batch_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(batch_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(batch_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c')
        pred = tf.nn.tanh(batch_norm(pred, 'g_pred_bn'))

        return pred

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
