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
import matplotlib
matplotlib.use('Agg')
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

print('Load dlib')
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../dlib_b/python_examples/shape_predictor_68_face_landmarks.dat')

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
plt.ioff()

print('Set Train Data')
# set train data
batch_size = 128

train_set = []

landmarks_frame = pd.read_csv('./face_landmarks_base.csv')
file_list = landmarks_frame.image_name.values.tolist()

avgP_container = np.load('f_avgP_list_base.npz')
emb_container = np.load('f_emb_list_base.npz')

for key in sorted(emb_container, key=lambda x: int(x.strip('arr_'))) :
    batch = avgP_container[key], emb_container[key]
    if len(batch[0]) == batch_size :
        train_set.append(batch)

t_dataset = helper.Dataset('nf',file_list, 160)


test_set = []

t_file_list = glob('./SNF_TESTSET/*')

t_avgP_container = np.load('f_avgP_list_wild.npz')
t_emb_container = np.load('f_emb_list_wild.npz')

for key in sorted(t_emb_container, key=lambda x: int(x.strip('arr_'))) :
    batch = t_avgP_container[key], t_emb_container[key]
    test_set.append(batch)

t_t_dataset = helper.Dataset('nf',t_file_list, 160)

print('Set Model')
# set model
def l_init() :
    return tf.contrib.layers.variance_scaling_initializer()

def F_layer(net, f_num = 1024, is_training=False) : 
    with tf.variable_scope('F') :
        net = slim.fully_connected(net, f_num, weights_initializer=l_init(),scope='fc0')
        net = slim.fully_connected(net, f_num, weights_initializer=l_init(),scope='fc1')
    return net

def MLP(net, landmark_num = 68, is_training=False, reuse=None, scope='MLP'):
    """Builds the MLP for landmark"""
    with tf.variable_scope(scope, 'MLP') :
        net = slim.fully_connected(net, 1024, weights_initializer=l_init(), scope='fc0')
        net = slim.fully_connected(net, 512, weights_initializer=l_init(), scope='fc1')
        net = slim.fully_connected(net, 256, weights_initializer=l_init(), scope='fc2')
        net = slim.fully_connected(net, 128, weights_initializer=l_init(), scope='fc3')
        net = slim.fully_connected(net, landmark_num, activation_fn=None, weights_initializer=l_init(), scope='fc4')
    return net

def landmark_decode(net, landmark_num = 68, is_training=False):
    with tf.variable_scope('landmark') :
        decoded_x = MLP(net, is_training=is_training, scope= 'decoded_x')
        decoded_y = MLP(net, is_training=is_training, scope= 'decoded_y')
    return decoded_x, decoded_y

def CNN(F, size) :
    with tf.variable_scope('CNN') :
        # 10 x 10 x 256
        with tf.variable_scope('features') :
            f_size = int(size / 16)
            features = slim.fully_connected(F, f_size * f_size * 256, weights_initializer= l_init(), activation_fn=None, scope="features")
            features = tf.reshape(features, [-1, f_size, f_size, 256])
            features = tf.nn.relu(features)
            print(features.shape)
            
        # 20 x 20 x 128
        with tf.variable_scope('upsample_0') :
            conv_0 = tf.layers.conv2d(features, 128, (3,3), padding='same', kernel_initializer= l_init(), activation=tf.nn.relu)
            conv_0 = tf.layers.conv2d(conv_0, 128, (3,3), padding='same', kernel_initializer= l_init(), activation=None)
            conv_0 = tf.nn.relu(conv_0 + features)
            print(conv_0.shape)
        
        # 40 x 40 x 64
        with tf.variable_scope('upsample_1') :
            f_size *= 2
            upsample_1 = tf.image.resize_nearest_neighbor(conv_0, (f_size, f_size))
            conv_1 = tf.layers.conv2d(upsample_1, 64, (3,3), padding='same', kernel_initializer= l_init(), activation=tf.nn.relu)
            conv_1 = tf.layers.conv2d(upsample_1, 64, (3,3), padding='same', kernel_initializer= l_init(), activation=None)
            
            upsample_1 = tf.layers.conv2d(upsample_1, 64, (1,1), padding='same', kernel_initializer= l_init(), activation=None)
            conv_1 = tf.nn.relu(conv_1 + upsample_1)
            print(conv_1.shape)
    
        # 80 x 80 x 32
        with tf.variable_scope('upsample_2') :
            f_size *= 2
            upsample_2 = tf.image.resize_nearest_neighbor(conv_1, (f_size, f_size))
            conv_2 = tf.layers.conv2d(upsample_2, 32, (3,3), padding='same', kernel_initializer= l_init(), activation=tf.nn.relu)
            conv_2 = tf.layers.conv2d(upsample_2, 32, (3,3), padding='same', kernel_initializer= l_init(), activation=None)
            
            upsample_2 = tf.layers.conv2d(upsample_2, 32, (1,1), padding='same', kernel_initializer= l_init(), activation=None)
            conv_2 = tf.nn.relu(conv_2 + upsample_2)
            print(conv_2.shape)
        
        # 160 x 160 x 32
        with tf.variable_scope('upsample_3') :
            f_size *= 2
            upsample_3 = tf.image.resize_nearest_neighbor(conv_2, (f_size, f_size))
            conv_3 = tf.layers.conv2d(upsample_3, 32, (3,3), padding='same', kernel_initializer= l_init(), activation=tf.nn.relu)
            conv_3 = tf.layers.conv2d(upsample_3, 32, (3,3), padding='same', kernel_initializer= l_init(), activation=None)
            
            upsample_3 = tf.layers.conv2d(upsample_3, 32, (1,1), padding='same', kernel_initializer= l_init(), activation=None)
            conv_3 = tf.nn.relu(conv_3 + upsample_3)
            print(conv_3.shape)
        
        with tf.variable_scope('one_by_one_conv') :
            one_by_one_conv = tf.layers.conv2d(conv_3, 3, (1,1), padding='same', kernel_initializer= l_init(), activation=None)
        
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

print('Set Hyperparam')
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
epochs = 5000
learning_rate = 1e-4
chk_interval = 100
log_interval = 8

avgP_num = 1792
emb_num = 128
f_num = 1792

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
    
    is_training = tf.placeholder(tf.bool)
    
    grids = tf.constant(xa, dtype=tf.float32, name= 'grids')
    grids_test = tf.constant(xa_t, dtype=tf.float32, name= 'grids')
    
    zero_displacement = (tf.constant(zd[:, :, 0], dtype=tf.float32, name= 'zd_x'), 
                         tf.constant(zd[:, :, 1], dtype=tf.float32, name= 'zd_y'))
    zero_displacement_test = (tf.constant(zd_t[:, :, 0], dtype=tf.float32, name= 'zd_x_t'), 
                              tf.constant(zd_t[:, :, 1], dtype=tf.float32, name= 'zd_y_t'))
    
    # model
    F = F_layer(avgP_inputs, f_num= f_num, is_training=is_training)
    
    (l_x_preds, l_y_preds) = landmark_decode(F, landmark_num= l_num, is_training=is_training)
    
    l_x_loss = tf.losses.mean_squared_error(l_x_labels, l_x_preds, reduction="weighted_mean", weights=1.0)
    l_y_loss = tf.losses.mean_squared_error(l_y_labels, l_y_preds, reduction="weighted_mean", weights=1.0)
    
    l_loss = tf.add(l_x_loss, l_y_loss)
    
    t_preds = texture_decode(F, t_size)
    t_loss = tf.losses.absolute_difference(t_labels, t_preds, weights=100.0)
    
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
    
    w_labels = tf.nn.l2_normalize(w_labels, 1)
    w_preds = tf.nn.l2_normalize(w_preds, 1)
    w_loss = tf.losses.cosine_distance(w_labels, w_preds, dim=1, weights=10.0)
    
    total_cost = l_loss + t_loss + w_loss
    opt = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    
    #train_writer = tf.summary.FileWriter(log_dir + '/board',sess.graph)
    
    sess.run(tf.global_variables_initializer())

def get_landmarks(n) :
    index = n[0] * batch_size + n[1]
    return landmarks_frame.ix[index]

def detect_landmarks(img, num_landmarks = 68) :
    scipy.misc.imsave('tmp.jpg',img)
    img = scipy.misc.imread('tmp.jpg')
    dets = detector(img, 1)  # face detection

    # ignore all the files with no or more than one faces detected.
    if len(dets) == 1:
        row = []

        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for i in range(num_landmarks):
            part_i_x = shape.part(i).x
            part_i_y = shape.part(i).y
            row += [[part_i_x, part_i_y]]
        return np.asarray(row, dtype=np.float32) 
    else :
        return None

def write_test_image(dir_name, e, t, w, o, test_t, test_w, test_o):
    if t is not None:
        t_img = scipy.misc.toimage(t)
        lt_file_name = dir_name + '/' + str(e) + '_lt.jpg'
        misc.imsave(lt_file_name, t_img)

    if w is not None:
        w_img = scipy.misc.toimage(w)
        w_file_name = dir_name + '/' + str(e) + '_w.jpg'
        misc.imsave(w_file_name, w_img)
    
    if o is not None:
        o_img = scipy.misc.toimage(o)
        o_file_name = dir_name + '/' + str(e) + '_o.jpg'
        misc.imsave(o_file_name, o_img)
    
    if test_t is not None:
        test_t_img = scipy.misc.toimage(test_t)
        test_t_file_name = dir_name + '/' + str(e) + '_test_t.jpg'
        misc.imsave(test_t_file_name, test_t_img)
        
    if test_w is not None:
        test_w_img = scipy.misc.toimage(test_w)
        test_w_file_name = dir_name + '/' + str(e) + '_test_w.jpg'
        misc.imsave(test_w_file_name, test_w_img)
        
    if test_o is not None:
        test_o_img = scipy.misc.toimage(test_o)
        test_o_file_name = dir_name + '/' + str(e) + '_test_o.jpg'
        misc.imsave(test_o_file_name, test_o_img)

print('start train')
with g.as_default():
    with sess.as_default() :
        start_test = time.time()
        
        for e in range(epochs):
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
                       opt]
                
                feed_dict = {avgP_inputs : f_avgP.reshape(-1, avgP_num),
                             l_x_labels : l_labels[:, :, 0].reshape(-1, l_num), 
                             l_y_labels : l_labels[:, :, 1].reshape(-1, l_num),
                             t_labels : t_label_batch, 
                             w_labels : f_emb.reshape(-1, emb_num), 
                             f_phase_train_placeholder:False,
                             is_training : True}

                out = sess.run(run, feed_dict= feed_dict)
                (l_x_cost, 
                 l_y_cost, 
                 t_cost, 
                 w_cost, 
                 _) = out
                
                l_x_cost_sum += l_x_cost
                l_y_cost_sum += l_y_cost
                t_cost_sum += t_cost
                w_cost_sum += w_cost

                if (i+1) % log_interval == 0 :
                    loss_log = "Iter: {}/{}".format(i+1, len(train_set))
                    loss_log += "Training loss: X = {:.4f}, Y = {:.4f}, T = {:.4f}, W = {:.4f}".format(l_x_cost_sum / log_interval, 
                                                                                                       l_y_cost_sum / log_interval, 
                                                                                                       t_cost_sum / log_interval, 
                                                                                                       w_cost_sum / log_interval)
                    print(loss_log)
                    f = open(loss_file_name, "a")
                    f.write(loss_log +'\n')
                    f.close
                    
                    l_x_cost_sum = 0
                    l_y_cost_sum = 0
                    t_cost_sum = 0
                    w_cost_sum = 0
        
                    test_index = (random.randint(0, len(train_set)-1),random.randint(0, len(train_set[0])-1))
                    test_avgP = train_set[test_index[0]][test_index[1]]
                    t_landmarks = get_landmarks(test_index)

                    t_img = misc.imread(t_landmarks[0])
                    t_l_labels = t_landmarks[1:].as_matrix().astype('float32').reshape(-1, 2)

                    test_run = [t_preds, warp_t]
                    test_feed = {avgP_inputs : test_avgP.reshape(-1, avgP_num),
                                 l_x_labels : t_l_labels[:, 0].reshape(-1, l_num), 
                                 l_y_labels : t_l_labels[:, 1].reshape(-1, l_num), 
                                 is_training : False}

                    t_t, t_w = sess.run(test_run, feed_dict= test_feed)

                    test_index = random.randint(0, len(test_set[0])-1)
                    test_avgP = test_set[0][test_index]
                    test_img = misc.imread(t_file_list[test_index])

                    test_run = t_preds
                    test_feed = {avgP_inputs : test_avgP.reshape(-1, avgP_num),
                                 is_training : False}

                    test_t = sess.run(test_run, feed_dict= test_feed)

                    dl = detect_landmarks(test_t.reshape(t_size, t_size, t_channel))

                    if dl is not None :
                        dl = dl.reshape(-1, 2)
                        test_run = warp_t
                        test_feed = {avgP_inputs : test_avgP.reshape(-1, avgP_num),
                                     l_x_labels : dl[:,0].reshape(-1, l_num), 
                                     l_y_labels : dl[:,1].reshape(-1, l_num)}

                        test_w = sess.run(test_run, feed_dict= test_feed)
                    else :
                        test_w = None

                    write_test_image(log_dir, e, 
                                     t_t.reshape(t_size, t_size, t_channel), 
                                     t_w.reshape(t_size, t_size, t_channel), 
                                     t_img.reshape(t_size, t_size, t_channel), 
                                     test_t.reshape(t_size, t_size, t_channel), 
                                     test_w.reshape(t_size, t_size, t_channel),
                                     test_img.reshape(t_size, t_size, t_channel))
                
            print("Epoch: {}/{}".format(e+1, epochs), "Time: %s" % (time.time() - start_test))
            
            chk_name = log_dir + "/chk/" + str(int((e+1)/chk_interval)) + "/nf.ckpt"
            chk_name = make_path(chk_name)
            saver = tf.train.Saver()
            save_path = saver.save(sess, chk_name)
            print("Model saved in file: %s" % save_path, "Time: %s" % (time.time() - start_test))