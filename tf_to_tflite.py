import tensorflow as tf
import config as config

cfg = config.cfg

model = tf.keras.models.load_model(cfg["model_path"], compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_quant_model = converter.convert()
open(cfg["tflite_model_path"], "wb").write(tflite_quant_model)
