import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import spikegan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='X_sample1.mat', help='path of the dataset')
parser.add_argument('--phase', dest='phase', default='train', help='path of the dataset')
parser.add_argument('--epochs', dest='epochs', default=100, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', default=16, help='number of epochs')
parser.add_argument('--display_interval', dest='display_interval', default=10, help='number of epochs')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint_dir/', help='number of epochs')
parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, help='number of epochs')
parser.add_argument('--num_examples_test', dest='num_examples_test', default=5000, help='number of epochs')

args = parser.parse_args()

def main(_):
    with tf.Session() as sess:
        model = spikegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    tf.app.run()
