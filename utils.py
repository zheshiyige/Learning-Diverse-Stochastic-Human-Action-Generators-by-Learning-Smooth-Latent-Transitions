# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os
import subprocess
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import viz
import cameras
import data_utils
import pdb

IMG_WIDTH = 320
IMG_HEIGHT = 240

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def video(sequence, tmp_dir, video_dir, name, data_mean_2d, data_std_2d, dim_to_ignore_2d, verbose=0, unnorm=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    if unnorm:
        sequence = data_utils.unNormalizeData(sequence, data_mean_2d, data_std_2d, dim_to_ignore_2d )
    else:
        sequence = data_utils.fillData(sequence, data_mean_2d, data_std_2d, dim_to_ignore_2d )

    gs1 = gridspec.GridSpec(1, 1)
    plt.axis('off')
    for t in range(sequence.shape[0]):
        # draw for sequence[t]
        plt.clf()

        ax2 = plt.subplot(gs1[0])
        p2d = sequence[t, :]
        viz.show2Dpose( p2d, ax2)
        ax2.invert_yaxis()
        if not os.path.exists(tmp_dir):
            os.system('mkdir -p "{}"'.format(tmp_dir))
        plt.savefig(os.path.join(tmp_dir, '%04d.jpg' %(t)))

    if not os.path.exists(video_dir):
        os.system('mkdir -p "{}"'.format(video_dir))
    np.save(os.path.join(video_dir, name), sequence)

    if verbose:
        os.system('ffmpeg -framerate 16 -y -i "' + os.path.join(tmp_dir, "%04d.jpg") + '" "' + os.path.join(video_dir, name + '.mp4"'))
    else:
        subprocess.call(['ffmpeg', '-framerate', '16', '-y', '-i', os.path.join(tmp_dir, "%04d.jpg"), os.path.join(video_dir, name + '.mp4')], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)


    os.system('rm '+os.path.join(tmp_dir, '*'))

def video_real(sequence, video_dir, name):
    if np.max(sequence) <= 1.01:
        outputdata = sequence * 255
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(video_dir, '{}.mp4'.format(name))
    skvideo.io.vwrite(file_path, outputdata, inputdict={'-r':'16'})

def draw_pose(pose, pose_dir, name, data_mean_2d, data_std_2d, dim_to_ignore_2d, verbose=0, unnorm=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    pose = np.expand_dims(pose, axis=0)

    if unnorm:
      pose = data_utils.fillData(pose, data_mean_2d, data_std_2d, dim_to_ignore_2d )
    else:
      pose = data_utils.unNormalizeData(pose, data_mean_2d, data_std_2d, dim_to_ignore_2d )


    #plt.figure()
    gs1 = gridspec.GridSpec(1, 1)
    plt.axis('off')
    for t in range(pose.shape[0]):
        # draw for sequence[t]
        plt.clf()

        ax2 = plt.subplot(gs1[0])
        p2d = pose[t, :]
        viz.show2Dpose( p2d, ax2)
        ax2.invert_yaxis()
        if not os.path.exists(pose_dir):
            os.system('mkdir -p "{}"'.format(pose_dir))
        plt.savefig(os.path.join(pose_dir, name), bbox_inches='tight', transparent=True, pad_inches=0)
 


def draw_seqpose(pose, pose_dir, name, data_mean_2d, data_std_2d, dim_to_ignore_2d, verbose=0, unnorm=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt



    if unnorm:
      pose = data_utils.fillData(pose, data_mean_2d, data_std_2d, dim_to_ignore_2d )
    else:
      pose = data_utils.unNormalizeData(pose, data_mean_2d, data_std_2d, dim_to_ignore_2d )


    #plt.figure()
    gs1 = gridspec.GridSpec(1, 1)
    plt.axis('off')
    
    # draw for sequence[t]
    plt.clf()
    ax2 = plt.subplot(gs1[0])
    viz.show2Dpose_seq( pose, ax2)
    ax2.invert_yaxis()
    if not os.path.exists(pose_dir):
          os.system('mkdir -p "{}"'.format(pose_dir))
    plt.savefig(os.path.join(pose_dir, name), bbox_inches='tight',  pad_inches=0, dpi=200)#transparent=True,

   


def imread(path):
    return scipy.misc.imread(path).astype(np.float)



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))



def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
