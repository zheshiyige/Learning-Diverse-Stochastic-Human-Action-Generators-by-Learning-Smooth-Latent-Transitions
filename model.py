from __future__ import division
from __future__ import print_function
import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from six.moves import xrange
import matplotlib
matplotlib.use('Agg')
from ops import *
from utils import *
import random
import pickle, h5py, cv2
import viz
import cameras
import data_utils
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fmin_l_bfgs_b
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

class PoseseqGAN(object):
    
    def __init__(self, sess, vec_length=32, seq_length=50, 
                 batch_size=64, sample_nbatch=20, actions = 'all',
                 first_z_dim=12, shift_z_dim=64,
                 checkpoint_dir=None, loader="saver", classes=10):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            first_z_dim: (optional) input dimension for single frame GAN input.
            shift_z_dim: (optional) input dimension for latent seq GAN . 
            seq_length: the number of action frames in training and testing, you can specify longer seq length than specified here.
            vec_length: the dimension of single frame action
            actions: which actions to train and sample from
            loader: mode to use model loader, "saver" : save and restore all network parameters, "loader": save and restore only the single frame generator network
            "tester": save and restore only latent action sequence generator network parameters. 
            classes: number of training and testing classes of actions
            checkpoint_dir: save the model to the directory 
        """
        self.sess = sess
        self.classes = classes
        self.batch_size = batch_size
        self.sample_size = batch_size
        self.image_shape = [seq_length, vec_length]
        self.first_z_dim = first_z_dim
        self.shift_z_dim = shift_z_dim
        self.val_nbatch = sample_nbatch
        self.k = 0.0001
        self.mag_reg = 0.1
        self.reg_class = 0.01
        self.checkpoint_dir = checkpoint_dir
        self.use_loader = loader
        self.build_model()
        self.actions = actions
        self.model_name = "PoseseqGAN.model"

    def h36m_data_loader(self, batch_size, dataset, sequence_length, actions):
        counter = 0
        batch = []
        class_batch = []
        while 1:
            for k in dataset.keys():
                if k[1] in actions:
                    class_vec = [1.0 if k[1] == action_name else 0.0 for action_name in actions]
                    video = dataset[k]
                    start = random.randint(300, video.shape[0]-1-self.image_shape[0]*3)
                    clip = video[start:start+self.image_shape[0]*3, :]
                    idxs = np.arange(0, self.image_shape[0]*3, 3)
                    clip = clip[idxs]

                    counter += 1
                    batch.append(clip)
                    class_batch.append(class_vec)

                    if counter >= batch_size:
                        batch = np.array(batch)
                        class_batch = np.array(class_batch)
                        yield batch, class_batch
                        counter = 0
                        batch = []
                        class_batch = []

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
      
        lstm_z = tf.placeholder(tf.float32, [None, self.image_shape[0], self.shift_z_dim]) #lstm input

        self.class_vec = tf.placeholder(tf.float32, [None, self.classes], name='class_one_hot')

        self.first_z = tf.placeholder(tf.float32, [None, self.first_z_dim], name='first_z')
        self.shift_z = tf.placeholder(tf.float32, [None, self.image_shape[0]-1, self.shift_z_dim], name='seq_z')

        self.kp = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.g_is_training = tf.placeholder(tf.bool, name='g_is_training')

        self.G_first, self.G, self.z_shifts, self.latent = self.rnn_generator(self.first_z, self.shift_z, self.class_vec)
        self.D, self.D_logits = self.discriminator(self.images, self.class_vec)

        self.D_, self.D_logits_ = self.discriminator(self.G, self.class_vec, reuse=True)
        self.z_shifts_sum = []

        self.d_real = tf.reduce_mean(self.D)
        self.d_fake = tf.reduce_mean(self.D_)

        # get l2 smoothness
        self.diff = []
        for i in range(self.image_shape[0] - 1):
            self.diff.append(tf.square(self.G[:, i, :, 0]- self.G[:, i+1, :, 0]))

        l2_diff = tf.stack(self.diff, axis=1)
        print(l2_diff.get_shape(), len(self.diff),self.G.get_shape(), '=======================================================')

        l2_diff = tf.reduce_sum(tf.reduce_sum(l2_diff, axis=2),  axis=1)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                  labels=tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                   labels=tf.zeros_like(self.D_)))

        self.g_gan_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                   labels=tf.ones_like(self.D_)))

        # Regularize z_shifts
        self.z_shifts_reg = tf.reduce_sum(tf.square(tf.stack(self.z_shifts, axis=1)))

        self.g_smooth_loss = tf.reduce_mean(l2_diff)

        ##############Bidirectional GAN and Cycle consistency loss#################

        #classify real action sequences
        self.pred = self.LSTM_classifier(self.images)
        self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.class_vec))
        correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.class_vec, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #classify generated action sequences
        self.G_pred = self.LSTM_classifier(self.G, reuse=True)
        self.pred_class = tf.argmax(self.G_pred, 1)
        self.G_pred_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.G_pred, labels=self.class_vec))
        print('cycle consistency loss')

        self.g_loss = self.g_gan_loss + self.k * self.g_smooth_loss + self.mag_reg * self.z_shifts_reg + self.reg_class*(self.class_loss + self.G_pred_class_loss)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/shifts_generator')
        self.global_step   = tf.placeholder(tf.int32, name="global_step")

        g_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/first_frame_generator')

        self.loader = tf.train.Saver(max_to_keep=200, var_list=g_vars)
        self.tester = tf.train.Saver(max_to_keep=200, var_list=self.g_vars)
        self.saver  = tf.train.Saver(max_to_keep=200)

    def LSTM_classifier(self, image, reuse=False):
        with tf.variable_scope("generator/shifts_generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h = bi_lstm_class(tf.reshape(image, [-1, self.image_shape[0], self.image_shape[1]]), n_steps=self.image_shape[0], n_input=self.image_shape[1],
            name='class_bi_lstm')

            return  h

    def train(self, config):

        batch_idxs = 500
        decay_rate = 0.5    
        epochs_to_decay = 30
        boundaries = []
        lr_values = [config.learning_rate]
        for exp in range(1, 6):
          lr_values.append(config.learning_rate * (decay_rate)**exp)
          boundaries.append(batch_idxs * exp * epochs_to_decay)

        print(boundaries, lr_values)
        self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, lr_values)
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        # TODO: check gradient magnitude here
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator/shifts_generator')
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)

            g_grad = opt.compute_gradients(self.g_loss, var_list=self.g_vars)
            g_optim = opt.minimize(self.g_loss, var_list=self.g_vars)

            if config.check_grad:
                layer_to_check = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=config.check_grad)
                check_list = []
                for pair in g_grad:
                    if pair[1] in layer_to_check:
                        check_list.append(pair[0])
                       
        self.sess.run( tf.global_variables_initializer() )

        counter = 1
        start_time = time.time()

        if config.load:
            print('self.checkpoint_dir', self.checkpoint_dir)
            load_model = self.load(self.checkpoint_dir)
            if load_model == True:
                print("""
                ======
                An existing model was found in the checkpoint directory.
                If you want to train a new model from scratch,
                delete the checkpoint directory or specify a different
                --checkpoint_dir argument.
                ======
                """)

            else:
                raise ValueError("""
                ======
                pretrained model not found, please train a single pose model first 
                ======
                """)
        else:
            print("""
            ======
            An existing model was not found in the checkpoint directory.
            Initializing a new one.
            ======
            """)

        # ======load data==================
        actions = data_utils.define_actions(self.actions)
        number_of_actions = len( actions )
        print('actions',actions)
        # Load camera parameters
        SUBJECT_IDS = [1,5,6,7,8,9,11]
        rcams = cameras.load_cameras(config.cameras_path, SUBJECT_IDS)
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data_corrected( actions, config.dataset, rcams )
        print( "done reading and normalizing data." )
        n = 0
        for key2d in train_set_2d.keys():
            n2d, _ = train_set_2d[ key2d ].shape
            n = n + n2d//self.image_shape[0]

        nbatches = n // config.batch_size
        tr_loader = self.h36m_data_loader(self.batch_size, train_set_2d, config.seq_length, actions)
        te_loader = self.h36m_data_loader(self.sample_size, test_set_2d, config.seq_length, actions)

        for epoch in xrange(config.epoch):
            for idx in xrange(0, batch_idxs):

                batch_images, batch_class = tr_loader.__next__()
                batch_pose = batch_images[:, 0, :]
                if config.check_input:
                    if epoch == 0 and idx == 0:
                        for i in range(batch_images.shape[0]):
                            video(batch_images[i], config.tmp_dir, os.path.join(config.video_dir, 'D_input', 'class_%02d' %int(np.where(batch_class[i])[0])), '%04d' %(i), data_mean_2d, data_std_2d, dim_to_ignore_2d)

                first_z = np.random.uniform(-1, 1, [config.batch_size, self.first_z_dim]) \
                            .astype(np.float32)
                shift_z = np.random.normal(0.0, 1.0, [config.batch_size, self.image_shape[0]-1, self.shift_z_dim ])

                for _ in range(1):

                    _ = self.sess.run([d_optim],
                        feed_dict={self.images: batch_images, self.first_z: first_z, self.shift_z: shift_z, self.kp: 0.5, self.class_vec: batch_class,
                                    self.global_step: counter, self.g_is_training: True})

                for _ in range(2):
                    # Update G network
                    _, g_loss = self.sess.run([g_optim, self.z_shifts_reg],
                        feed_dict={ self.images: batch_images, self.first_z: first_z, self.shift_z: shift_z, self.kp: 0.5 , self.class_vec: batch_class,
                                    self.global_step: counter, self.g_is_training: True})
                   
                print('\n', g_loss)

                if config.check_grad:
                    grads = self.sess.run(check_list,
                        feed_dict={ self.first_z: first_z, self.shift_z: shift_z, self.kp: 0.5 ,
                                    self.global_step: counter, self.g_is_training: True})
                    for grad in grads:
                        print('\n', grad)

                errD_fake, errD_real, errG = \
                self.sess.run([self.d_loss_fake, self.d_loss_real, self.g_loss],
                               {self.first_z: first_z, self.shift_z: shift_z, self.images: batch_images, self.kp: 0.5, self.class_vec: batch_class,
                                self.global_step: counter, self.g_is_training: True})

                counter += 1
                print("\rEpoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG), end='')

                #if np.mod(counter, 10) == 1:
                if np.mod(counter, 2*batch_idxs) == 1:
                    d_loss_s, g_loss_s, d_real_s, d_fake_s = 0.0, 0.0, 0.0, 0.0
                    class_counter = [0] * self.classes

                    for b in range(self.val_nbatch):

                        first_s_z = np.random.uniform(-1, 1, size=(self.sample_size , self.first_z_dim))
                        shift_s_z = np.random.normal(0.0, 1.0, size=(self.sample_size, self.image_shape[0]-1, self.shift_z_dim))

                        sample_images, sample_class = te_loader.__next__()
                        sample_pose = sample_images[:, 0, :]
                      
                        samples, d_loss, g_loss, d_real, d_fake = self.sess.run(
                            [self.G, self.d_loss, self.g_loss, self.d_real, self.d_fake],
                            feed_dict={self.first_z: first_s_z, self.shift_z: shift_s_z, self.images: sample_images, self.kp: 1.0, self.class_vec: sample_class,
                                       self.g_is_training: False}
                        )

                        d_loss_s += d_loss
                        g_loss_s += g_loss
                        d_real_s += d_real
                        d_fake_s += d_fake
                        if config.val_save:

                            for i in range(samples.shape[0]//4):
                                class_num = int(np.where(sample_class[i])[0])
                                video(np.squeeze(samples[i]), config.tmp_dir, os.path.join(config.video_dir, 'train_%02d_%04d' %(epoch, idx), 'class_%02d' %class_num), '%04d' %(class_counter[class_num]), data_mean_2d, data_std_2d, dim_to_ignore_2d)
                                class_counter[class_num] += 1

                    d_loss_s /= self.val_nbatch
                    g_loss_s /= self.val_nbatch
                    d_real_s /= self.val_nbatch
                    d_fake_s /= self.val_nbatch
                    print("\n[Sample] d_loss: %.8f, g_loss: %.8f, d_real: %.8f, d_fake: %.8f" % (d_loss_s, g_loss_s, d_real_s, d_fake_s))

                if np.mod(counter, batch_idxs) == 1:
                    self.save(config.checkpoint_dir, counter + config.global_counter)

    def test(self, config):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter
        from sklearn import manifold, datasets
        
        batch_idxs = 500
        counter = 1
        start_time = time.time()
        actions = ["Directions","Discussion","Eating","Greeting",
            "Phoning", "Posing", "Sitting", "SittingDown",
            "Smoking", "Walking"]
        load_model = self.load(self.checkpoint_dir)
        if load_model == True:
                print("""
                ======
                An existing model was found in the checkpoint directory.
                If you want to train a new model from scratch,
                delete the checkpoint directory or specify a different
                --checkpoint_dir argument.
                ======

                """)
        else:
                raise ValueError("""
                ======
                pretrained model not found, please train a single pose model first 
                ======
                """)
        
        # ======load data==================
        actions = data_utils.define_actions(self.actions)
        number_of_actions = len(actions)
        # Load camera parameters
        SUBJECT_IDS = [1,5,6,7,8,9,11]
        rcams = cameras.load_cameras(config.cameras_path, SUBJECT_IDS)
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data_corrected( actions, config.dataset, rcams )
        print( "done reading and normalizing data." )
        n = 0
        for key2d in train_set_2d.keys():
            n2d, _ = train_set_2d[ key2d ].shape
            n = n + n2d//self.image_shape[0]

        nbatches = n // config.batch_size
        sample_size = 10000
        tr_loader = self.h36m_data_loader(sample_size, train_set_2d, config.seq_length, actions)
        te_loader = self.h36m_data_loader(sample_size, test_set_2d, config.seq_length, actions)
        colors  = ['r', 'g', 'b', 'y', 'm', 'c', 'k', (0,0.9,0.93), (0.2,1.0,0.2), (0.38, 0.69, 1.0)]
             
        if config.testmode == 'all_class':
              class_counter = [0] * self.classes
              sample_class = np.identity(self.classes)
              sample_class = np.tile(sample_class, (int(self.batch_size/self.classes), 1))

              for val_idx in range(self.val_nbatch):
                        first_s_z = np.random.uniform(-1, 1, size=(self.batch_size, self.first_z_dim))
                        shift_s_z = np.random.normal(0.0, 1.0, size=(self.batch_size, self.image_shape[0]-1, self.shift_z_dim))
                        samples = self.sess.run(
                            self.G,
                            feed_dict={self.first_z: first_s_z, self.shift_z: shift_s_z, self.kp: 1.0, self.class_vec: sample_class,
                                       self.g_is_training: False}
                        )
                        for i in range(samples.shape[0]):
                                print('save video')
                                class_num = int(np.where(sample_class[i])[0])
                                video(np.squeeze(samples[i]), config.tmp_dir, os.path.join(config.video_dir, 'test_%04d' %(val_idx), 'class_%02d' %class_num), '%04d' %(class_counter[class_num]), data_mean_2d, data_std_2d, dim_to_ignore_2d)
                                draw_seqpose(np.squeeze(samples[i]), os.path.join(config.pose_dir, 'test_%04d' %(val_idx), 'class_%02d' %class_num), '%02d.png'%(i), data_mean_2d, data_std_2d, dim_to_ignore_2d)
                                plt.close('all')
                                class_counter[class_num] += 1


    def discriminator(self, image, class_vec, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h = bidirectional_lstm(tf.reshape(image, [-1, self.image_shape[0], self.image_shape[1]]), class_vec, n_steps=self.image_shape[0], n_input=self.image_shape[1], name='d_bi_lstm')
            return tf.nn.sigmoid(h), h

    def rnn_generator(self, first_z, seq_z, class_vec, n_hidden=256, mlp=False):
        print('new one layer 1024 rnn_generator')
        with tf.variable_scope("generator"):

            ###################
            # Generate 1st frame
            z_base = first_z
            z_shift = seq_z

            with tf.variable_scope('first_frame_generator'):

                h0, h0_w, h0_b = linear(tf.concat([z_base, class_vec], axis=1), 1024, 'g_h0_lin', with_w=True)
                h6 = tf.nn.relu(batch_norm(h0, decay=0.99, scale=True, is_training=self.g_is_training, scope='g_bn0'))
    
                # This is the generated first frame
                h7, h7_w, h7_b = linear(h6, self.image_shape[1], 'g_h7_lin', with_w=True)
                first_frame = h7
                frames = [first_frame]

            with tf.variable_scope('shifts_generator'):

                # Generate the shifts
                class_rnn = tf.expand_dims(class_vec, 1)
                class_rnn = tf.tile(class_rnn, [1, self.image_shape[0]-1, 1])
           
                lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
                init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, tf.concat([z_shift, class_rnn], axis=2), initial_state=init_state, time_major=False)

                l_out_x = tf.reshape(outputs, [-1, n_hidden], name='rnn_out_reshape')
                hn9, hn9_w, hn9_b = linear(l_out_x, self.first_z_dim, 'g_n_h9_lin', with_w=True)
                hn9 = tf.reshape(hn9, [-1, (self.image_shape[0]-1)*self.first_z_dim])
                z_list = tf.split(tf.clip_by_value(hn9, -1, 1), self.image_shape[0]-1, 1)
  
            latent = []
            with tf.variable_scope("first_frame_generator", reuse=True):

                z_current = z_base
                latent.append(z_current)
                for i in range(self.image_shape[0] - 1):
                    z_current = z_current + z_list[i]
                    latent.append(z_current)
                    h0, h0_w, h0_b = linear(tf.concat([z_current, class_vec], axis=1), 1024, 'g_h0_lin', with_w=True)
                    h6 = tf.nn.relu(batch_norm(h0, decay=0.99, scale=True, is_training=self.g_is_training, scope='g_bn0'))

                    # This is the generated first frame
                    h7, h7_w, h7_b = linear(h6, self.image_shape[1], 'g_h7_lin', with_w=True)
                    first_frame = h7
                    frames.append(h7)
            return first_frame, tf.expand_dims(tf.stack(frames, axis=1), -1), z_list, tf.stack(latent, axis=1)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print('ckpt.model_checkpoint_path',  ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            if self.use_loader == "saver":
                #self.loader.restore(self.sess, ckpt.model_checkpoint_path)
                print('self.loader loaded the model')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.use_loader == "loader":
                self.loader.restore(self.sess, ckpt.model_checkpoint_path)

            if self.use_loader == "tester":
                print('using tester')
                self.loader.restore(self.sess, ckpt.model_checkpoint_path)
                self.tester.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
