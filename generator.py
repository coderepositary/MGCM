# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow
#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random
import pickle

from layerscgan import *

img_height = 64
img_width = 64
output_size = 64
output_c_dim=3
img_layer = 3
img_size = img_height * img_width

ngf = 64
ndf = 64

def discriminator(batch_size,input, image,name="discriminator"):

    with tf.variable_scope(name) as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     assert tf.get_variable_scope().reuse == False

        dis_input = tf.concat([image, input], 3)
        h0 = lrelu(conv2d(dis_input, ndf, name='d_h0_conv'))
        # h0 is (32 x 32 x self.df_dim)
        h1 = lrelu(batch_norm((conv2d(h0, ndf*2, name='d_h1_conv')),name='d_bn1'))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu(batch_norm((conv2d(h1, ndf*4, name='d_h2_conv')),name='d_bn2'))
        # h2 is (8x 8 x self.df_dim*4)
        h3 = lrelu(batch_norm((conv2d(h2, ndf*8, d_h=1, d_w=1, name='d_h3_conv')),name='d_bn3'))
        # h3 is (4 x 4 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

def generator(batch_size,image,text_image,name="generator"):
    with tf.variable_scope(name) as scope:

        s = output_size
        s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)

        # image is (64 x 64 x input_c_dim)
        e1 = conv2d(image, ndf, name='g_e1_conv')
        # e1 is (32 x 32 x self.gf_dim)
        e2 = batch_norm(conv2d(lrelu(e1), ndf*2, name='g_e2_conv'),name='g_bn_e2')
        # e2 is (16 x 16 x self.gf_dim*2)
        e3 = batch_norm((conv2d(lrelu(e2), ndf*4, name='g_e3_conv')),name='g_bn_e3')
        # e3 is (8 x 8 x self.gf_dim*4)
        e4 = batch_norm((conv2d(lrelu(e3), ndf*8, name='g_e4_conv')),name='g_bn_e4')
        # e4 is (4 x 4 x self.gf_dim*8)
        e5 = batch_norm((conv2d(lrelu(e4), ndf*8, name='g_e5_conv')),name='g_bn_e5')
        # e5 is (2 x 2 x self.gf_dim*8)
        # e6 = batch_norm((conv2d(lrelu(e5), ndf*8, name='g_e6_conv')),name='g_bn_e6')
        # # e6 is (4 x 4 x self.gf_dim*8)
        # e7 = batch_norm((conv2d(lrelu(e6), ndf*8, name='g_e7_conv')),name='g_bn_e7')
        # # e7 is (2 x 2 x self.gf_dim*8)
        e6 = batch_norm((conv2d(lrelu(e5), ndf*8, name='g_e8_conv')),name='g_bn_e8')
        e6 = tf.reshape(e6, [-1,ndf*8])
        e7 = tf.concat([e6,text_image],1)
        e6_W = tf.get_variable('e6_W', [e7.shape[1],ndf*8],dtype=tf.float32,trainable=True)
        e8 = tf.reshape(tf.sigmoid(tf.matmul(e7,e6_W)),[-1,1,1,ndf*8])

        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8),
            [batch_size, s32, s32, ndf*8], name='g_d1', with_w=True)
        d11 = tf.nn.dropout(batch_norm((d1),name='g_bn_d1'), 0.5)
        d1 = tf.concat([d1, e5], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
            [batch_size, s16, s16, ndf*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(batch_norm((d2),name='g_bn_d2'), 0.5)
        d2 = tf.concat([d2, e4], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
            [batch_size, s8, s8, ndf*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(batch_norm((d3),name='g_bn_d3'), 0.5)
        d3 = tf.concat([d3, e3], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
            [batch_size, s4, s4, ndf*8], name='g_d4', with_w=True)
        d4 = batch_norm((d4),name='g_bn_d4')
        d4 = tf.concat([d4, e2], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
            [batch_size, s2, s2, ndf*4], name='g_d5', with_w=True)
        d5 = batch_norm((d5),name='g_bn_d5')
        d5 = tf.concat([d5, e1], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        # d6, d6_w, d6_b = deconv2d(tf.nn.relu(d5),
        #     [batch_size, s4, s4, ndf*2], name='g_d6', with_w=True)
        # d6 = batch_norm((d6),name='g_bn_d6')
        # d6 = tf.concat([d6, e2], 3)
        # # d6 is (64 x 64 x self.gf_dim*2*2)
        #
        # d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
        #     [batch_size, s2, s2, ndf], name='g_d7', with_w=True)
        # d7 = batch_norm((d7),name='g_bn_d7')
        # d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d5),
            [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return e5,d11,tf.nn.tanh(d8)


def build_encoder(batch_size,image, name="encoder"):
    with tf.variable_scope(name) as scope:

        s = output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (64 x 64 x input_c_dim)
        e1 = conv2d(image, ndf, name='g_e1_conv')
        # e1 is (32 x 32 x self.gf_dim)
        e2 = batch_norm((conv2d(lrelu(e1), ndf*2, name='g_e2_conv')),name='g_bn_e2')
        # e2 is (16 x 16 x self.gf_dim*2)
        e3 = batch_norm((conv2d(lrelu(e2), ndf*4, name='g_e3_conv')),name='g_bn_e3')
        # e3 is (8 x 8 x self.gf_dim*4)
        e4 = batch_norm((conv2d(lrelu(e3), ndf*8, name='g_e4_conv')),name='g_bn_e4')
        # e4 is (4 x 4 x self.gf_dim*8)
        e5 = batch_norm((conv2d(lrelu(e4), ndf*8, name='g_e5_conv')),name='g_bn_e5')
        # e5 is (2 x 2 x self.gf_dim*8)

        return e5