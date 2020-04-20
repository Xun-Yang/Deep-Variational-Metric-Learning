from __future__ import absolute_import
from datasets import data_provider
from lib import GoogleNet_Model, Losses, nn_Ops, evaluation
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import tensorflow as tf
import os
import time
import numpy as np
import keras.backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

############## HYPERPARAMETERS ###############################

BATCH_SIZE=128
DATASET= 'cars196'
IMAGE_SIZE=227
EMBEDDING_SIZE=128
LR_init=7e-5
LR_gen=1e-2
LR_s=1e-3
bp_epoch=200
MAX_ITER=1200

#############################################################
image_mean = np.array([123, 117, 104], dtype=np.float32)  # RGB
# To shape the array image_mean to (1, 1, 1, 3) => three channels
image_mean = image_mean[None, None, None, [2, 1, 0]]

neighbours = [1, 2, 4, 8, 16, 32]
products_neighbours = [1, 10, 1000]
############## DATASET GENERATOR #############################
streams = data_provider.get_streams(BATCH_SIZE, DATASET, "n_pairs_mc", crop_size=IMAGE_SIZE)
stream_train, stream_train_eval, stream_test = streams


LOGDIR = './tensorboard_log/'+DATASET+'/'+time.strftime('%m-%d-%H-%M', time.localtime(time.time()))+'/'
nn_Ops.create_path(time.strftime('%m-%d-%H-%M', time.localtime(time.time())))



tfd = tfp.distributions
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(EMBEDDING_SIZE), scale=1),
                        reinterpreted_batch_ndims=1)

def mysampling(z_mean, z_log_var):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def main(_):
    x_raw = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('isphase'):
        is_Phase = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)
    with tf.name_scope('lambdas'):
        lambda1 = tf.placeholder(tf.float32)
        lambda2 = tf.placeholder(tf.float32)
        lambda3 = tf.placeholder(tf.float32)
        lambda4 = tf.placeholder(tf.float32)

    with tf.variable_scope('Feature_extractor'):
        google_net_model = GoogleNet_Model.GoogleNet_Model()
        embedding = google_net_model.forward(x_raw)
        embedding_y_origin = embedding
        # output
        embedding = nn_Ops.bn_block(
            embedding, normal=True, is_Training=is_Training, name='FC3')
        embedding_z = nn_Ops.fc_block(
            embedding, in_d=1024, out_d=EMBEDDING_SIZE,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)
        # predict mu
        embedding1 = nn_Ops.bn_block(
            embedding, normal=True, is_Training=is_Training, name='FC1')
        embedding_mu = nn_Ops.fc_block(
            embedding1, in_d=1024, out_d=EMBEDDING_SIZE,
            name='fc2', is_bn=False, is_relu=False, is_Training=is_Training)
        # predict (log (sigma^2))
        embedding2 = nn_Ops.bn_block(
            embedding, normal=True, is_Training=is_Training, name='FC2')
        embedding_sigma = nn_Ops.fc_block(
            embedding2, in_d=1024, out_d=EMBEDDING_SIZE,
            name='fc3', is_bn=False, is_relu=False, is_Training=is_Training)


        
        with tf.name_scope('Loss'):
            def exclude_batch_norm(name):
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            wdLoss = 5e-3 * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            label = tf.reduce_mean(label_raw, axis=1)
            J_m = Losses.triplet_semihard_loss(label,embedding_z)+ wdLoss
            


    zv_emb = mysampling(embedding_mu, embedding_sigma)
    # zv = tfd.Independent(tfd.Normal(loc=embedding_mu, scale=embedding_sigma),
                        # reinterpreted_batch_ndims=1)
    # zv_emb=zv.sample([1])
    zv_emb1=tf.reshape(zv_emb,(-1,128))
    # zv_prob=zv.prob(zv_emb)
    # prior_prob=zv.prob(zv_emb)

    embedding_z_add = tf.add(embedding_z, zv_emb1,name='Synthesized_features')
    with tf.variable_scope('Decoder'):
        embedding_y_add = nn_Ops.fc_block(
            embedding_z_add, in_d=EMBEDDING_SIZE, out_d=512,
            name='decoder1', is_bn=True, is_relu=True, is_Training=is_Phase
        )
        embedding_y_add = nn_Ops.fc_block(
            embedding_y_add, in_d=512, out_d=1024,
            name='decoder2', is_bn=False, is_relu=False, is_Training=is_Phase
        )
    print("embedding_sigma",embedding_sigma)
    print("embedding_mu",embedding_mu)
    with tf.name_scope('Loss_KL'):
    	kl_loss = 1 + embedding_sigma - K.square(embedding_mu) - K.exp(embedding_sigma)
    	kl_loss = K.sum(kl_loss, axis=-1)
    	kl_loss *= -0.5
        # L1 loss
    	J_KL = lambda1 * K.mean(kl_loss) 

    with tf.name_scope('Loss_Recon'):

        # L2 Loss
        J_recon = lambda2 * (0.5) * tf.reduce_sum(tf.square(embedding_y_add - embedding_y_origin))

    with tf.name_scope('Loss_synthetic'):

        # L3 Loss
        J_syn = lambda3 * Losses.triplet_semihard_loss(labels=label,embeddings=embedding_z_add)

    with tf.name_scope('Loss_metric'):

        # L4 Loss
        J_metric = lambda4 * J_m 

    with tf.name_scope('Loss_Softmax_classifier'):
        cross_entropy, W_fc, b_fc = Losses.cross_entropy(embedding=embedding_y_origin, label=label)
    
