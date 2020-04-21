from __future__ import absolute_import
from datasets import data_provider
from lib import GoogleNet_Model, Losses, nn_Ops, evaluation
import copy
from tensorflow.contrib import layers
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import tensorflow as tf
import os
import time
import numpy as np
import keras.backend as K
from tqdm import tqdm
from parameters import *

# A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

# Creating Streams from the dataset
streams = data_provider.get_streams(BATCH_SIZE, DATASET, crop_size=IMAGE_SIZE)
stream_train, stream_train_eval, stream_test = streams

LEN_TRAIN = stream_train.data_stream.dataset.num_examples
MAX_ITER = int(LEN_TRAIN/BATCH_SIZE)

# check system time
_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
LOGDIR = './tensorboard_log/'+DATASET+'/'+_time+'/'
nn_Ops.create_path(_time)

# tfd = tfp.distributions
# prior = tfd.Independent(tfd.Normal(loc=tf.zeros(EMBEDDING_SIZE), scale=1),reinterpreted_batch_ndims=1)

def samplingGaussian(z_mean, z_log_var):
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

    # placeholders

    # -> raw input data
    x_raw = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])

    # -> parameters involved
    with tf.name_scope('isTraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('isPhase'):
        is_Phase = tf.placeholder(tf.bool)
    with tf.name_scope('learningRate'):
        lr = tf.placeholder(tf.float32)
    with tf.name_scope('lambdas'):
        lambda1 = tf.placeholder(tf.float32)
        lambda2 = tf.placeholder(tf.float32)
        lambda3 = tf.placeholder(tf.float32)
        lambda4 = tf.placeholder(tf.float32)

    # -> feature extractor layers
    with tf.variable_scope('FeatureExtractor'):

        # modified googLeNet layer pre-trained on ILSVRC2012
        google_net_model = GoogleNet_Model.GoogleNet_Model()
        embedding_gn = google_net_model.forward(x_raw)

        # batch normalization of average pooling layer of googLeNet
        embedding = nn_Ops.bn_block(embedding_gn, normal=True, is_Training=is_Training, name='BN')

        # 3 fully connected layers of Size EMBEDDING_SIZE
        # mean of cluster
        embedding_mu = nn_Ops.fc_block(embedding, in_d=1024, out_d=EMBEDDING_SIZE,name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)
        # log(sigma^2) of cluster
        embedding_sigma = nn_Ops.fc_block(embedding, in_d=1024, out_d=EMBEDDING_SIZE,name='fc2', is_bn=False, is_relu=False, is_Training=is_Training)
        # invariant feature of cluster
        embedding_zi = nn_Ops.fc_block(embedding, in_d=1024, out_d=EMBEDDING_SIZE,name='fc3', is_bn=False, is_relu=False, is_Training=is_Training)
        
        with tf.name_scope('Loss'):
            def exclude_batch_norm(name):
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            wdLoss = 5e-3 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)])
            label = tf.reduce_mean(label_raw, axis=1)
            J_m = Losses.triplet_semihard_loss(label,embedding_zi)+ wdLoss
    
    # Generator
    with tf.variable_scope('Generator'):
        embedding_re = samplingGaussian(embedding_mu, embedding_sigma)
        embedding_zv = tf.reshape(embedding_re,(-1,EMBEDDING_SIZE))
        # Z = Zi + Zv
        embedding_z = tf.add(embedding_zi, embedding_zv, name='Synthesized_features')

    # Decoder
    with tf.variable_scope('Decoder'):
        embedding_y1 = nn_Ops.fc_block(embedding_z, in_d=EMBEDDING_SIZE, out_d=512,name='decoder1', is_bn=True, is_relu=True, is_Training=is_Phase)
        embedding_y2 = nn_Ops.fc_block(embedding_y1, in_d=512, out_d=1024,name='decoder2', is_bn=False, is_relu=False, is_Training=is_Phase)

    print("embedding_sigma",embedding_sigma)
    print("embedding_mu",embedding_mu)

    # Defining the 4 Losses

    # L1 loss
    # Definition: L1 = (1/2) x sum( 1 + log(sigma^2) - mu^2 - sigma^2)
    with tf.name_scope('L1_KLDivergence'):
    	kl_loss = 1 + embedding_sigma - K.square(embedding_mu) - K.exp(embedding_sigma)
    	kl_loss = K.sum(kl_loss, axis=-1)
    	kl_loss *= -(0.5/BATCH_SIZE)
    	L1 = lambda1 * K.mean(kl_loss) 

    # L2 Loss
    # Definition: L2 = sum( L2Norm( target - outputOf(GoogleNet) ))
    with tf.name_scope('L2_Reconstruction'):
        L2 = lambda2 * (1/(20*BATCH_SIZE)) * tf.reduce_sum(tf.square(embedding_y2 - embedding_gn))

    # L3 Loss
    # Definition: L3 = Lm( Z )
    with tf.name_scope('L3_Synthesized'):
        L3 = lambda3 * Losses.triplet_semihard_loss( labels=label, embeddings=embedding_z)

    # L4 Loss
    # Definition: L4 = Lm( Zi )
    with tf.name_scope('L4_Metric'):
        L4 = lambda4 * J_m

    # Classifier Loss
    with tf.name_scope('Softmax_Loss'):
        cross_entropy, W_fc, b_fc = Losses.cross_entropy(embedding=embedding_gn, label=label)
    
    
    c_train_step = nn_Ops.training(loss= L4 + L3 + L1, lr=lr, var_scope='FeatureExtractor')
    g_train_step = nn_Ops.training(loss=L2, lr=LR_gen, var_scope='Decoder')
    s_train_step = nn_Ops.training(loss=cross_entropy, lr=LR_s, var_scope='Softmax_classifier')

    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    model_summary()

    with tf.Session(config=config) as sess:

        summary_writer = tf.summary.FileWriter(LOGDIR,sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        _lr = LR_init

        # To Record the losses to TfBoard
        Jm_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)
        L1_loss = nn_Ops.data_collector(tag='KLDivergence', init=1e+6)
        L2_loss = nn_Ops.data_collector(tag='Reconstruction', init=1e+6)
        L3_loss = nn_Ops.data_collector(tag='Synthesized', init=1e+6)
        L4_loss = nn_Ops.data_collector(tag='Metric', init=1e+6)
        
        cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
        wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)
        max_nmi = 0

        print("Phase 1")
        step = 0
        epoch_iterator = stream_train.get_epoch_iterator()
        for epoch in tqdm(range(NUM_EPOCHS_PHASE1)):
            print("Epoch: ",epoch)
            for batch in tqdm(copy.copy(epoch_iterator),total=MAX_ITER):
                step += 1
                # get images and labels from batch
                x_batch_data, Label_raw = nn_Ops.batch_data(batch)

                # training step
                c_train, g_train, s_train, wd_Loss_var, L1_var,L4_var, J_m_var, \
                    L3_var, L2_var, cross_en_var = sess.run(
                        [c_train_step, g_train_step, s_train_step, wdLoss, L1,
                         L4, J_m, L3, L2, cross_entropy],
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True,is_Phase:False, 
                                   lambda1:1, lambda2:1, lambda3:0.1, lambda4:1,  lr: _lr})
                
                Jm_loss.update(var=J_m_var)
                L1_loss.update(var=L1_var)
                L2_loss.update(var=L2_var)
                L3_loss.update(var=L3_var)
                L4_loss.update(var=L4_var)
                cross_entropy_loss.update(cross_en_var)
                wd_Loss.update(var=wd_Loss_var)

                # evaluation
                if step % RECORD_STEP == 0:
                    nmi_te, f1_te, recalls_te = evaluation.Evaluation(stream_test, image_mean, sess, x_raw, label_raw, is_Training,is_Phase, embedding_zi, 98, neighbours)

                    # Summary
                    eval_summary = tf.Summary()
                    eval_summary.value.add(tag='test nmi', simple_value=nmi_te)
                    eval_summary.value.add(tag='test f1', simple_value=f1_te)
                    for i in range(0, np.shape(neighbours)[0]):
                        eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                    Jm_loss.write_to_tfboard(eval_summary)
                    eval_summary.value.add(tag='learning_rate', simple_value=_lr)
                    L1_loss.write_to_tfboard(eval_summary)
                    L2_loss.write_to_tfboard(eval_summary)
                    L3_loss.write_to_tfboard(eval_summary)
                    L4_loss.write_to_tfboard(eval_summary)
                    wd_Loss.write_to_tfboard(eval_summary)
                    cross_entropy_loss.write_to_tfboard(eval_summary)
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary Recorder')
                    if nmi_te > max_nmi:
                        max_nmi = nmi_te
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                        print("Model Saved")
                    summary_writer.flush()    

        
        print("Phase 2")
        epoch_iterator = stream_train.get_epoch_iterator()
        for epoch in tqdm(range(NUM_EPOCHS_PHASE2)):
            print("Epoch: ",epoch)
            for batch in tqdm(copy.copy(epoch_iterator),total=MAX_ITER):
                step += 1
                # get images and labels from batch
                x_batch_data, Label_raw = nn_Ops.batch_data(batch)

                # training step
                c_train, g_train, s_train, wd_Loss_var, L1_var,L4_var, J_m_var, \
                    L3_var, L2_var, cross_en_var = sess.run(
                        [c_train_step, g_train_step, s_train_step, wdLoss, L1,
                         L4, J_m, L3, L2, cross_entropy],
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True,is_Phase:True,
                                   lambda1:0.8, lambda2:1, lambda3:0.2, lambda4:0.8, lr: _lr})
                
                Jm_loss.update(var=J_m_var)
                L1_loss.update(var=L1_var)
                L2_loss.update(var=L2_var)
                L3_loss.update(var=L3_var)
                L4_loss.update(var=L4_var)
                wd_Loss.update(var=wd_Loss_var)
                cross_entropy_loss.update(cross_en_var)

                # evaluation
                if step % RECORD_STEP == 0:
                    nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                        stream_test, image_mean, sess, x_raw, label_raw, is_Training,is_Phase, embedding_zi, 98, neighbours)

                    # Summary
                    eval_summary = tf.Summary()
                    eval_summary.value.add(tag='test nmi', simple_value=nmi_te)
                    eval_summary.value.add(tag='test f1', simple_value=f1_te)
                    for i in range(0, np.shape(neighbours)[0]):
                        eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                    Jm_loss.write_to_tfboard(eval_summary)
                    eval_summary.value.add(tag='learning_rate', simple_value=_lr)
                    L1_loss.write_to_tfboard(eval_summary)
                    L2_loss.write_to_tfboard(eval_summary)
                    L3_loss.write_to_tfboard(eval_summary)
                    L4_loss.write_to_tfboard(eval_summary)
                    wd_Loss.write_to_tfboard(eval_summary)
                    cross_entropy_loss.write_to_tfboard(eval_summary)
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary Recorder')
                    if nmi_te > max_nmi:
                        max_nmi = nmi_te
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                        print("Model Saved")
                    summary_writer.flush()

if __name__ == '__main__':
    tf.app.run()