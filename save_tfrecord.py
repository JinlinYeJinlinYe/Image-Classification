import tensorflow as tf
import config as config
from tqdm import tqdm
import os

cfg = config.cfg

writer = tf.io.TFRecordWriter(cfg["tfrecord_file"])

for index, name in enumerate(os.listdir(cfg["datas_path"])):  # enumerate(): 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # print('index', index, name)
    class_path = cfg["datas_path"] + name + '/'
    for img_name in tqdm(os.listdir(class_path)):  # os.listdir: 返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        img_path = class_path + img_name
        image = open(img_path, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())  # 将一个example写入TFRecord文件
writer.close()
