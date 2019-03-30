import os
import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
import PIL.Image
from alexnet import AlexNet

num_bit = 48
num_class = 196
keep_prob = 1
img_size = 227
checkpoint = "../model/ISDH-48b/"
model_name = "ISDH-48b"
skip_layers = ['fc8']
mean_value = np.array([123, 117, 104]).reshape((1, 3))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
config.device_count['GPU'] = 0
config.gpu_options.allow_growth = True


def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    print(numOfImage, bit_length)
    string = ''
    for i in range(bit_length):
        string += '0' if binary_like_values[0][i] <= 0 else '1'
    string_list = [int(i) for i in string]
    return string_list


def evaluate(img_path):
    image = tf.placeholder(tf.float32, [1, img_size, img_size, 3], name='image')
    model = AlexNet(image, keep_prob, num_bit, num_class, skip_layers)

    D = model.softsign

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        checkpoint = "../models/ISDH-48b/"
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        print('Restoring model from {}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        print('Loading success')
        data = np.zeros([1, img_size, img_size, 3], np.float32)
        img = PIL.Image.open(img_path)
        img = img.resize((img_size, img_size))
        new_im = img - mean_value
        new_im = new_im.astype(np.int16)
        data = np.array(new_im).reshape(1, img_size, img_size, 3)
        eval_sess = sess.run(D, feed_dict={image: data})
        w_res = toBinaryString(eval_sess)
        print(w_res)


img_path = sys.argv[1]
evaluate(img_path)
