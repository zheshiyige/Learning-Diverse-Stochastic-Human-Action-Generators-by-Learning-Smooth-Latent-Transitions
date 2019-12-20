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
import viz
import cameras
import data_utils
import pdb

class poseWGAN(object):
    # sample_size = 64
    def __init__(self, sess, vec_length=32, seq_length=50, 
                 batch_size=64, sample_size=128, actions = 'all',
                 z_dim=12, 
                 checkpoint_dir=None, loader="saver", lam=10, classes=10):
        """
       Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) input dimension for single frame GAN input.
            seq_length: the number of action frames in training and testing, you can specify longer seq length than specified here.
            vec_length: the dimension of single frame action
            actions: which actions to train and sample from
            loader: mode to use model loader, "saver" : save and restore all network parameters, "loader": save and restore only the single frame generator network and discriminator network parameters.
            classes: number of training and testing classes of actions
            checkpoint_dir: save the model to the directory 
        """
        self.sess = sess
        self.classes = classes
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_shape = [seq_length, vec_length]
        self.z_dim = z_dim
        self.lam = 10
        self.val_nbatch = 20

        self.checkpoint_dir = checkpoint_dir
        self.use_loader = loader
        self.build_model()
        self.actions = actions

        self.model_name = "poseWGAN.model"

    def h36m_data_loader(self, batch_size, dataset, sequence_length, actions):
        counter = 0
        batch = []
        class_batch = []

        while 1:
            for k in dataset.keys():
                if k[1] in actions:
                    # class one-hot vec
                    class_vec = [1.0 if k[1] == action_name else 0.0 for action_name in actions]
                    video = dataset[k]
                    start = random.randint(300, video.shape[0]-1)
                    clip = video[start, :]

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
       
        self.pose_vec = tf.placeholder(tf.float32, [None, self.image_shape[1]], name='real_pose_vec')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.class_vec = tf.placeholder(tf.float32, [None, self.classes], name='class_one_hot')
        self.kp = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.p_kp = tf.placeholder(tf.float32, name='P_dropout_keep_prob')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.g_is_training = tf.placeholder(tf.bool, name='g_is_training')
        self.p_is_training = tf.placeholder(tf.bool, name='p_is_training')
        self.G_first = self.generator(self.z, self.class_vec)
        with tf.name_scope("p_real") as scope:
            self.P_logits = self.pose_discriminator(self.pose_vec, self.class_vec)
            self.p_real_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

        with tf.name_scope("p_fake") as scope:
            self.P_logits_ = self.pose_discriminator(self.G_first, self.class_vec, reuse=True)
            self.p_fake_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)


        self.p_real = tf.reduce_mean(self.P_logits)
        self.p_fake = tf.reduce_mean(self.P_logits_)

        self.p_loss = tf.reduce_mean(self.P_logits_ - self.P_logits)

        alpha_dist = tf.contrib.distributions.Uniform(0., 1.)
        alpha = alpha_dist.sample((tf.shape(self.pose_vec)[0], 1))
        interpolated = self.pose_vec + alpha*(self.G_first-self.pose_vec)
        inte_logit = self.pose_discriminator(interpolated, self.class_vec, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        # gradient_penalty = tf.reduce_mean(tf.nn.relu(grad_l2-1))
        gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
        self.p_gp_loss_sum = tf.summary.scalar("p_gp_loss", gradient_penalty)
        self.p_grad_norm = tf.summary.scalar("p_grad_norm", tf.nn.l2_loss(gradients))
        self.p_loss += self.lam*gradient_penalty

        self.g_pose_loss = - tf.reduce_mean(self.P_logits_)
        self.g_loss = self.g_pose_loss


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.p_loss_sum = tf.summary.scalar("p_loss", self.p_loss)

        t_vars = tf.trainable_variables()


        self.p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/first_frame_generator')
      

        self.p_vars_sum = []
        for p_v in self.p_vars:
            self.p_vars_sum.append(tf.summary.histogram(p_v.name, p_v))

      
        self.global_step   = tf.placeholder(tf.int32, name="global_step")

        g_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/first_frame_generator')
        p_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_discriminator')
        self.loader = tf.train.Saver(max_to_keep=200, var_list=g_vars+p_vars)
        self.saver  = tf.train.Saver(max_to_keep=200)



    def train(self, config):

        # Define some parameters
        batch_idxs = 500
        decay_rate = 0.5     # empirical
        epochs_to_decay = 50

        boundaries = []
        lr_values = [config.learning_rate]
        for exp in range(1, 6):
          lr_values.append(config.learning_rate * (decay_rate)**exp)
          boundaries.append(batch_idxs * exp * epochs_to_decay)

        print(boundaries, lr_values)
        self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, lr_values)
        self.lr_sum = tf.summary.scalar("lr", self.learning_rate)


        # TODO: check gradient magnitude here
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator/first_frame_generator')
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)

            g_grad = opt.compute_gradients(self.g_loss, var_list=self.g_vars)
            g_optim = opt.minimize(self.g_loss, var_list=self.g_vars)
            grad_sum = []

            if config.check_grad:

                layer_to_check = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=config.check_grad)

                check_list = []

                for pair in g_grad:
                    if pair[1] in layer_to_check:
                        check_list.append(pair[0])
                        grad_sum.append(tf.summary.histogram(pair[0].name, pair[0]))



        p_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9) \
                          .minimize(self.p_loss, var_list=self.p_vars)


        self.sess.run( tf.global_variables_initializer() )

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.g_loss_sum] + grad_sum)
        self.p_sum = tf.summary.merge(
            [self.z_sum, self.p_loss_sum, self.p_grad_norm, self.p_gp_loss_sum] + self.p_vars_sum)


        counter = 1
        start_time = time.time()

        if config.load:
            self.load(self.checkpoint_dir)
            print("""

            ======
            An existing model was found in the checkpoint directory.
            If you want to train a new model from scratch,
            delete the checkpoint directory or specify a different
            --checkpoint_dir argument.
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

        tr_loader = self.h36m_data_loader(config.batch_size, train_set_2d, config.seq_length, actions)
        te_loader = self.h36m_data_loader(config.sample_size, test_set_2d, config.seq_length, actions)

        for epoch in xrange(config.epoch):

            for idx in xrange(0, batch_idxs):


                batch_pose, batch_class = tr_loader.__next__()
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if config.check_input:

                    if epoch == 0 and idx == 0:
                        for i in range(batch_pose.shape[0]):
                            draw_pose(batch_pose[i], os.path.join(config.pose_dir, 'input', 'class_%02d' %int(np.where(batch_class[i])[0])), '%d.jpg' %(i), data_mean_2d, data_std_2d, dim_to_ignore_2d)

              

                if epoch < 25:
                    n_iter = 25
                else:
                    n_iter = 5

                for _ in range(5):

                    _, p_loss = self.sess.run([p_optim, self.p_loss],
                        feed_dict={ self.z: batch_z, self.kp: 0.5, self.p_kp: 0.5, self.pose_vec: batch_pose, self.class_vec: batch_class,
                                    self.global_step: counter, self.g_is_training: True})

                for _ in range(1):

                    # Update G network
                    _, g_loss = self.sess.run([g_optim, self.g_loss],
                        feed_dict={ self.z: batch_z, self.kp: 0.5 , self.p_kp: 0.5, self.class_vec: batch_class,
                                    self.global_step: counter, self.g_is_training: True})
                   

                errG, errP, g_sum_str, p_sum_str, lr_sum = \
                self.sess.run([self.g_loss, self.p_loss,
                               self.g_sum, self.p_sum, self.lr_sum],
                               {self.z: batch_z, self.pose_vec: batch_pose, self.kp: 1.0, self.p_kp: 1.0, self.class_vec: batch_class,
                                self.global_step: counter, self.g_is_training: False})

                counter += 1
                print("\rEpoch: [%2d] [%4d/%4d] time: %4.4f, p_loss: %.8f,  g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errP, errG), end='')


                if np.mod(counter, batch_idxs) == 1:

                    g_loss_s = 0.0
                    p_loss_s, p_real_s, p_fake_s = 0.0, 0.0, 0.0
                    for b in range(self.val_nbatch):

                        sample_z = np.random.uniform(-1, 1, size=(config.sample_size , self.z_dim))
                        sample_pose, sample_class = te_loader.__next__()

                        samples, g_loss, p_loss = self.sess.run(
                            [self.G_first, self.g_loss, self.p_loss],
                            feed_dict={self.z: sample_z, self.kp: 1.0, self.p_kp: 1.0, self.pose_vec: sample_pose, self.class_vec: sample_class,
                                       self.g_is_training: False}
                        )


                        g_loss_s += g_loss
                        p_loss_s += p_loss

                        if config.val_save:
                            if b >= 1:
                                continue

                            for i in range(samples.shape[0]):
    
                                draw_pose(samples[i], os.path.join(config.pose_dir, 'train_%02d_%04d' %(epoch, idx), 'class_%02d' %int(np.where(sample_class[i])[0])), 'recons_%d.jpg' %(i), data_mean_2d, data_std_2d, dim_to_ignore_2d)
                          


                    g_loss_s /= self.val_nbatch
                    p_loss_s /= self.val_nbatch

                    print("\n[Sample] g_loss: %.8f, p_loss: %.8f" % (g_loss_s, p_loss))

                if np.mod(counter, 4000) == 1:
                    self.save(config.checkpoint_dir, counter + config.global_counter)



    def pose_discriminator(self, pose_vec, class_vec, reuse=False):

        pose_vec = tf.concat([pose_vec, class_vec], axis=1)

        with tf.variable_scope("pose_discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()


            self.h0, self.h0_w, self.h0_b = linear(pose_vec, 1024, 'p_h0_lin', with_w=True)
            h6 = tf.nn.relu(self.h0)

            self.h7, self.h7_w, self.h7_b = linear(h6, 1, 'p_h7_lin', with_w=True)
            return self.h7


    def generator(self, z, class_vec, mlp=False):

       
        with tf.variable_scope("generator"):
            
            z = tf.concat([z, class_vec], axis=1)

            with tf.variable_scope('first_frame_generator'):

                h0, h0_w, h0_b = linear(z, 1024, 'g_h0_lin', with_w=True)
                h1 = tf.nn.relu(batch_norm(h0, decay=0.99, scale=True, is_training=self.g_is_training, scope='g_bn0'))

                
                # This is the generated first frame
                h1, h1_w, h1_b = linear(h1, self.image_shape[1], 'g_h7_lin', with_w=True)
                first_frame = h1


            return first_frame



    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if self.use_loader == "saver":
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.use_loader == "loader":
                self.loader.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


