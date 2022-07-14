import tensorflow as tf
import numpy as np
import cv2
import os
import keras
from os import listdir
from os.path import join
import matplotlib.pyplot as plt

def use_model(path_image, base_model):
    width = 128
    base_model = tf.keras.models.load_model(base_model)
    rimg = []
    predict_images = {}
    for item in listdir(path_image):
        file = join(path_image, item)
        # if item.split(".")[0] != "":
        img = cv2.imread(file, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, width))
        rimg = np.array(img)
        rimg = rimg.astype("float32")
        rimg /= 255
        rimg = np.reshape(rimg, (1, 128, 128, 3))
        predict = base_model.predict(rimg)
        label = ["true", "false"]
        result = label[np.argmax(predict)]
        predict_images[file] = str(result)

        # cvt_image = cv2.imread(file)
        # cvt_image = cv2.resize(cvt_image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        # cvt_image = cv2.cvtColor(cvt_image, cv2.COLOR_BGRA2RGBA)
        # plt.imshow(cvt_image)
        # result_txt = str(result)
        # if result_txt == 'true':
        #     plt.title("Transfer Slip")
        # else:
        #     plt.title("Not Transfer Slip")
        # plt.xlabel(item) 
        # plt.show()
        
    print("use model succeed")
    return predict_images
