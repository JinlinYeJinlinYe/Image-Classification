labels = {
    "garbage_labels":["厨余垃圾","有害垃圾","可回收垃圾","其他垃圾","背景图"],
}

cfg = {
    "datas_path": './dataset/garbage_sorting/',
    "tfrecord_file": "./dataset/garbage_sorting_train.tfrecord",
    "tflite_model_path": "./models/garbage_sorting_model.tflite",
    "model_path": "./models/garbage_sorting_model.h5",
    "labels_list": labels["garbage_labels"],
    "camera_id": 0,

    "width": 224,
    "height": 224,
    "color_channel": 3,

    "batch_size": 32,
    "epoch": 1 ,
    "lr": 1e-2,
    "save_freq": 1,
}
