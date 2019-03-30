import tensorflow as tf
from PIL import Image
import numpy as np
import json

img_dir = './data/myself/train_img/'
image_size = 227

# img
f1 = open('./data/myself/train_img_196.txt')
imagelists = f1.readlines()
l = len(imagelists)

# label
pid_label_file = './data/myself/feature_label.json'
with open(pid_label_file) as fl:
    line = fl.readline().strip('\n')
    pid_label = json.loads(line)

train_label = []
for img_name in imagelists:
    img_pid = img_name.split('_')[0]
    img_name_label = pid_label[img_pid]
    train_label.append(img_name_label)

train_label = np.array(train_label, dtype='uint8')

writer = tf.python_io.TFRecordWriter("train-myself.tfrecords")

for i in np.arange(l):
    img_name = imagelists[i].strip('\n\r')
    img_path = img_dir + img_name
    img = Image.open(img_path)
    img = img.resize((image_size, image_size))
    new_im = np.array(img)
    new_im = new_im / 255.0
    new_im = new_im.astype(np.int16)
    img_raw = new_im.tobytes()
    feature = train_label[i,:]
    print(img_name)
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
writer.close()
