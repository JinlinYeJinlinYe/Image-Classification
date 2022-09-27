import tensorflow as tf
import config as config
import cv2

cfg = config.cfg

def getDataset(tfrecord_file=cfg["tfrecord_file"]):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码JPEG图片
        return feature_dict['image'], feature_dict['label']

    # 数据预处理
    def preprocess(x, y):
        """
        x is a simple image, not a batch
        """
        x = tf.cast(x, dtype=tf.float32)
        x = tf.image.resize(x, [224, 224])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
        # x = tf.expand_dims(x, 0)
        x /= 255.0  # 归一化到[0,1]范围
        y = tf.cast(y, dtype=tf.int32)
        # y = tf.one_hot(y, depth=6)
        return x, y

    def normal(img, xy):
        _mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        _std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        img = tf.cast(img, dtype=tf.float32)
        img = img - _mean / _std
        return img, xy

    raw_dataset = raw_dataset.map(_parse_example)
    raw_dataset = raw_dataset.map(preprocess)
    # raw_dataset = raw_dataset.map(normal)
    return raw_dataset

def main():
    dataset = getDataset()

    for image, label in dataset:
        # label = np.asarray(label, np.int32)
        print("image.shape", image.shape,  "label", label.shape, type(label))
        image = image.numpy()
        # print(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        print("lable", label)
        # print("label", cfg["labels_list"][label])
        cv2.waitKey(100)

if __name__ == "__main__":
    main()


