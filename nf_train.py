from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import time
import random
from glob import glob
from datetime import datetime

import align.detect_face
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import misc
from scipy.interpolate import Rbf
from skimage import io, transform
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

import facenet
import helper

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
plt.ion()

print('set train data')
# set train data
batch_size = 100

landmarks_frame = pd.read_csv('./face_landmarks.csv')
file_list = landmarks_frame.image_name.values.tolist()

avgP_container = np.load('f_avgP_list.npz')
emb_container = np.load('f_emb_list.npz')

train_set = []
test_set = []

for key in sorted(emb_container, key=lambda x: int(x.strip('arr_'))) :
    batch = avgP_container[key], emb_container[key]
    if len(batch[0]) == batch_size :
        train_set.append(batch)
    else :
        test_set.append(batch)

test_set = train_set[-1]
train_set = train_set[0:-1]

t_dataset = helper.Dataset('nf',file_list, 160)

print('set model')
# set model
def F_layer(encoded, f_num = 128) : 
    with tf.variable_scope('F') :
        fc = slim.fully_connected(encoded, f_num, activation_fn=tf.nn.relu, scope='fc')
    return fc

def MLP(net, landmark_num = 68, reuse=None, scope='MLP'):
    """Builds the MLP for landmark"""
    with tf.variable_scope(scope, 'MLP') :
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='fc0')
        net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.fully_connected(net, landmark_num, activation_fn=tf.nn.relu, scope='fc2')
    return net

def landmark_decode(net, landmark_num = 68):
    with tf.variable_scope('landmark') :
        decoded_x = MLP(net, scope= 'decoded_x')
        decoded_y = MLP(net, scope= 'decoded_y')
    return decoded_x, decoded_y

def CNN(F, size) :
    with tf.variable_scope('CNN') :
        # 12 x 12 x 256
        f_size = int(size / 8)
        features = slim.fully_connected(F, f_size * f_size * 256, activation_fn=None, scope="features")
        features = tf.reshape(features, [-1, f_size, f_size, 256])
        # print(features.shape)
        
        # 24 x 24 x 128
        upsample_0 = slim.conv2d_transpose(features, 128, 5, stride=2, scope="upsample_0")
        # print(upsample_0.shape)
        
        # 48 x 48 x 64
        upsample_1 = slim.conv2d_transpose(upsample_0, 64, 5, stride=2, scope="upsample_1")
        # print(upsample_1.shape)
        
        # 96 x 96 x 32
        upsample_2 = slim.conv2d_transpose(upsample_1, 32, 5, stride=2, scope="upsample_2")
        # print(upsample_2.shape)
        
        # 96 x 96 x 3
        one_by_one_conv = slim.conv2d(upsample_2, 3, 1, stride=1, activation_fn=None, scope="one_by_one_conv")
        # print(one_by_one_conv.shape)
    return one_by_one_conv

def texture_decode(net, size) :
    with tf.variable_scope('texture') :
        cnn = CNN(net, size)
    return cnn

def get_grids(size):
    return np.mgrid[0:size-1:(size * 1j), 0:size-1:(size * 1j)]

def get_zero_displacement(size):
    mid = size/2
    end = size-1
    
    zero_displacement = [[0,0], 
                         [0, mid], 
                         [0, end], 
                         [mid, 0], 
                         [end,0], 
                         [end, mid], 
                         [end, end], 
                         [mid, end]]
    return zero_displacement

def rbf_tf(pred_x, pred_y, correct_points, grids, grid_shape):
    def _euclidean_norm_tf(x1, x2):
        euclidean_norm = tf.subtract(x1, x2)
        euclidean_norm = tf.square(euclidean_norm)
        euclidean_norm = tf.reduce_sum(euclidean_norm, 1)
        euclidean_norm = tf.add(euclidean_norm, 1e-10)
        #euclidean_norm = tf.clip_by_value(euclidean_norm, 0.1, 10**5)
        euclidean_norm = tf.sqrt(euclidean_norm)
        return euclidean_norm
        
        with tf.variable_scope('euclidean_norm') :
            euclidean_norm = tf.sqrt(tf.reduce_sum(((x1 - x2)**2), 1))
        return euclidean_norm

    def _h_linear_tf(r):
        return r

    def _call_norm_tf(x1, x2):
        with tf.variable_scope('norm') :
            x1 = tf.expand_dims(x1, 3)
            x2 = tf.expand_dims(x2, 2)
            n = norm(x1, x2)
        return n

    # set parameters
    norm = _euclidean_norm_tf
    basis_function = _h_linear_tf
    epsilon = tf.constant(2.)
    smooth = tf.constant(1.)

    xi = tf.stack([pred_x, pred_y], axis= 1)
    N = xi.shape[-1].value # same as landmarks_num => 76
    di = tf.expand_dims(correct_points, 2) # (None, 76, 1)
    
    r = _call_norm_tf(xi, xi) # (None, 76, 76)
    
    batch_shape = tf.shape(pred_x)[0:1]
    A = tf.subtract(basis_function(r), tf.multiply(smooth, tf.eye(N, batch_shape= batch_shape)))
    
    nodes = tf.matrix_solve (A, di)
    r2 = _call_norm_tf(grids, xi)
    return tf.reshape(tf.matmul(r2, nodes), [-1, grid_shape[0], grid_shape[1]])

def warp_tf(data, pred_x, pred_y, correct_x, correct_y, grids, grid_shape, zero_displacement) :
    with tf.variable_scope('pred') :
        pred_x_zd = tf.concat([pred_x, zero_displacement[0]], axis=1)
        pred_y_zd = tf.concat([pred_y, zero_displacement[1]], axis=1)
    with tf.variable_scope('correct') :
        correct_x_zd = tf.concat([correct_x, zero_displacement[0]], axis=1)
        correct_y_zd = tf.concat([correct_y, zero_displacement[1]], axis=1)

    with tf.variable_scope('rbf_x') :
        rbf_x = rbf_tf(pred_x_zd, pred_y_zd, correct_x_zd, grids, grid_shape)
    with tf.variable_scope('rbf_y') :
        rbf_y = rbf_tf(pred_x_zd, pred_y_zd, correct_y_zd, grids, grid_shape)

    with tf.variable_scope('resample') :
        warp = tf.stack([rbf_x, rbf_y], axis= 3)
        resample = tf.contrib.resampler.resampler(data=data, warp=warp)
    return resample

print('set hyperparam')
# summary
base_dir = os.path.expanduser('logs')
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(base_dir, subdir)
loss_file_name = log_dir + "/loss.txt"

def make_path(file_name) :
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return file_name

print(make_path(loss_file_name))

# hyperparam
epochs = 300

avgP_num = 1792
emb_num = 128
f_num = 1024

l_num = 68
zd_l_num = 76
t_size = 160
t_channel = 3

grid_y, grid_x = get_grids(t_size)
grid_shape = grid_x.shape

_xa = np.asarray([a.flatten() for a in [grid_x, grid_y]], dtype=np.float32) # (2, 25600)
xa = np.asarray([_xa for _ in range(0, batch_size)], dtype=np.float32) # (batch_size, 2, 25600)
xa_t = np.asarray([_xa], dtype=np.float32)

_zd = get_zero_displacement(t_size)
zd = np.asarray([_zd for _ in range(0, batch_size)], dtype=np.float32)
zd_t = np.asarray([_zd], dtype=np.float32)

g = tf.Graph()
with g.as_default():
    global_step = tf.Variable(0, trainable=False)
    
    # placeholder
    avgP_inputs = tf.placeholder(tf.float32, (None, avgP_num), name='avgP_inputs')
    
    l_x_labels = tf.placeholder(tf.float32, (None, l_num), name='l_x_labels')
    l_y_labels = tf.placeholder(tf.float32, (None, l_num), name='l_y_labels')
    t_labels = tf.placeholder(tf.float32, (None, t_size, t_size, t_channel), name='t_labels')
    w_labels = tf.placeholder(tf.float32, shape=(None, emb_num), name= 'w_labels')
    
    grids = tf.constant(xa, dtype=tf.float32, name= 'grids')
    grids_test = tf.constant(xa_t, dtype=tf.float32, name= 'grids')
    
    zero_displacement = (tf.constant(zd[:, :, 0], dtype=tf.float32, name= 'zd_x'), 
                         tf.constant(zd[:, :, 1], dtype=tf.float32, name= 'zd_y'))
    zero_displacement_test = (tf.constant(zd_t[:, :, 0], dtype=tf.float32, name= 'zd_x_t'), 
                              tf.constant(zd_t[:, :, 1], dtype=tf.float32, name= 'zd_y_t'))
    
    # model
    F = F_layer(avgP_inputs, f_num= f_num)
    
    (l_x_preds, l_y_preds) = landmark_decode(F, landmark_num= l_num)
    
    l_x_loss = tf.losses.mean_squared_error(l_x_labels, l_x_preds, reduction="weighted_mean")
    l_y_loss = tf.losses.mean_squared_error(l_y_labels, l_y_preds, reduction="weighted_mean")
    
    l_loss = tf.add(l_x_loss, l_y_loss)
    
    t_preds = texture_decode(F, t_size)
    t_loss = tf.losses.absolute_difference(t_labels, t_preds)
    
    with tf.variable_scope('warp') :
        warp = warp_tf(t_preds, 
                       l_x_preds, 
                       l_y_preds, 
                       l_x_labels, 
                       l_y_labels, 
                       grids, 
                       grid_shape, 
                       zero_displacement)
        
    with tf.variable_scope('warp_t') :
        warp_t = warp_tf(t_preds, 
                         l_x_preds, 
                         l_y_preds, 
                         l_x_labels, 
                         l_y_labels, 
                         grids_test, 
                         grid_shape, 
                         zero_displacement_test)
        warp_t = tf.cast(warp_t, tf.uint8)

with g.as_default():
    time_load_data = time.time()
    
    #facenet
    start_load_facenet = time.time()
    print("--- %s start load facenet ---" % (start_load_facenet))
    facenet.load_model('./20171012', input_map={"input:0": warp})
    print("--- %s facenet loaded ---" % (time.time() - start_load_facenet))

    f_phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    w_preds = tf.get_default_graph().get_tensor_by_name("embeddings:0")    
    w_loss = tf.losses.cosine_distance(w_labels, w_preds, dim=1)
    
    total_cost = l_loss + t_loss + w_loss
    
    opt = tf.train.AdamOptimizer(0.0001)# .minimize(total_cost)
    
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
    
    grads = tf.gradients(total_cost, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = opt.apply_gradients(grads_and_vars=grads)
    
    for grad, var in grads:
        tf.summary.histogram(var.op.name + '/gradient', grad)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)    
    
    sess.run(tf.global_variables_initializer())

def write_test_image(dir_name, e, l_x, l_y, t, w):
    t_img = scipy.misc.toimage(t)
    lt_file_name = dir_name + '/' + str(e) + '_lt.jpg'
    w_file_name = dir_name + '/' + str(e) + '_w.jpg'
    
    plt.figure()
    plt.imshow(t)
    plt.scatter(l_x, l_y, s=10, marker='.', c='b')
    plt.savefig(lt_file_name)
    
    plt.imshow(w)
    plt.savefig(w_file_name)

print('start train')
with g.as_default():
    with sess.as_default() :
        start_test = time.time()
        
        for e in range(epochs):
            log_index = 1
            l_x_cost_sum = 0
            l_y_cost_sum = 0
            t_cost_sum = 0
            w_cost_sum = 0
            
            for i, ((f_avgP, f_emb), t_label_batch) in enumerate(zip(train_set, t_dataset.get_batches(batch_size))):
                start = i * batch_size
                end = min(start+batch_size, len(train_set) * batch_size)
                size = end - start

                l_labels = landmarks_frame.ix[start:end - 1, 1:].as_matrix().astype('float32').reshape(size, l_num, 2)
                
                run = [l_x_loss, 
                       l_y_loss, 
                       t_loss, 
                       w_loss,
                       apply_grads, l_x_preds, l_y_preds]
                
                feed_dict = {avgP_inputs : f_avgP.reshape(-1, avgP_num),
                             l_x_labels : l_labels[:, :, 0].reshape(-1, l_num), 
                             l_y_labels : l_labels[:, :, 1].reshape(-1, l_num),
                             t_labels : t_label_batch, 
                             w_labels : f_emb.reshape(-1, emb_num), 
                             f_phase_train_placeholder:False}

                out = sess.run(run, feed_dict= feed_dict)
                (l_x_cost, 
                 l_y_cost, 
                 t_cost, 
                 w_cost, 
                 _, _l_x, _l_y) = out
                
                l_x_cost_sum += l_x_cost
                l_y_cost_sum += l_y_cost
                t_cost_sum += t_cost
                w_cost_sum += w_cost

                if (i+1) % log_index == 0 :
                    loss_log = "Iter: {}/{}".format(i+1, len(train_set))
                    loss_log += "Training loss: X = {:.4f}, Y = {:.4f}, T = {:.4f}, W = {:.4f}".format(l_x_cost_sum / log_index, l_y_cost_sum / log_index, t_cost_sum / log_index, w_cost_sum / log_index)
                    print(loss_log)
                    f = open(loss_file_name, "w")
                    f.write(loss_log +'\n')
                    f.close

                    l_x_cost_sum = 0
                    l_y_cost_sum = 0
                    t_cost_sum = 0
                    w_cost_sum = 0


            test_index = random.randint(0, len(test_set[0])-1)
            test_avgP = test_set[0][test_index] 
            t_landmarks = get_landmarks(test_index)

            t_img = t_landmarks[0]
            t_l_labels = t_landmarks[1:].as_matrix().astype('float32').reshape(-1, 2)

            test_run = [l_x_preds, l_y_preds, t_preds, warp_t]
            test_feed = {avgP_inputs : test_avgP.reshape(-1, avgP_num),
                         l_x_labels : t_l_labels[:, 0].reshape(-1, l_num), 
                         l_y_labels : t_l_labels[:, 1].reshape(-1, l_num)}

            t_l_x, t_l_y, t_t, t_w = sess.run(test_run, feed_dict= test_feed)
            write_test_image(log_dir, e, 
                            t_l_x.reshape(l_num),
                            t_l_y.reshape(l_num), 
                            t_t.reshape(t_size, t_size, t_channel), 
                            t_w.reshape(t_size, t_size, t_channel))
                
            print("Epoch: {}/{}".format(e+1, epochs), "Time: %s" % (time.time() - start_test))
            

            chk_name = "./chk/" + str(int((e+1)/50)) + "/nf.ckpt"
            chk_name = make_path(chk_name)
            saver = tf.train.Saver()
            save_path = saver.save(sess, chk_name)
            print("Model saved in file: %s" % save_path, "Time: %s" % (time.time() - start_test))