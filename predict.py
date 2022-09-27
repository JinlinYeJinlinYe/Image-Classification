import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
import cv2
import config as config


cfg = config.cfg

font_path = 'simsun.ttc'

# 在图像中显示中文
def putText(img, text, org=(0, 0), color=(0, 0, 255), font_size=80):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, fill=color, font=font)
    img = np.array(img_pil)
    return img

def loadModel(model_path=cfg["model_path"]):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def do_predict(model, img):
    img_src = cv2.resize(img, (cfg["height"], cfg["width"]))
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    image = img / 255.0
    image = image.reshape(1, cfg["height"], cfg["width"], cfg["color_channel"])
    result = model.predict(image)[0]
    lable_index = np.argmax(result)
    return lable_index

def main():
    cap = cv2.VideoCapture(cfg["camera_id"])
    model = loadModel()
    while True:
        _, img = cap.read()
        if _:
            label = do_predict(model, img)
            out_img = putText(img, cfg["labels_list"][label])
            out_img = cv2.resize(out_img, (320, 280))
            cv.imshow('predict', out_img)
            cv2.waitKey(10)

if __name__ == '__main__':
    main()