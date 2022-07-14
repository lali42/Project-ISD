import numpy as np
import re
import os
from os import listdir
import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"INS\libary\TesseractOCR\tesseract.exe"

def convert_img(path, folder, text_not_slip, text_slip):
    predict_images = {}
    for i in folder:
        for item in listdir(path):
            file = os.path.join(path, item)
            file_image, predict = search_text_transfer(file, file, text_not_slip, text_slip)
            cvt_image = cv2.imread(file)
            cvt_image = cv2.resize(cvt_image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            cvt_image = cv2.cvtColor(cvt_image, cv2.COLOR_BGRA2RGBA)
            if predict == "false":
                image = cv2.imread(file)
                list_color = [
                    (np.array([0, 0, 0]), np.array([160, 160, 160])),
                    (
                        np.array([0, 0, 0]),
                        np.array([170, 170, 170]),
                    ),
                ]
                for low, high in list_color:
                    mask = cv2.inRange(image, low, high)
                    image = cv2.bitwise_and(image, image, mask=mask)
                    file_image, predict = search_text_transfer(
                        file, image, text_not_slip, text_slip
                    )
                    if predict == "true":
                        break
                predict_images[file_image] = predict
            else:
                predict_images[file_image] = predict
            
            # cvt_image = cv2.imread(file_image)
            # cvt_image = cv2.resize(cvt_image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            # cvt_image = cv2.cvtColor(cvt_image, cv2.COLOR_BGRA2RGBA)
            # plt.imshow(cvt_image)
            # if predict == 'true':
            #     plt.title("Transfer Slip")
            # else:
            #     plt.title("Not Transfer Slip")
            # plt.xlabel(item)
            # plt.show()

    return predict_images

def search_text_transfer(file, image, text_not_slip, text_slip):
    predict = ""
    result = pytesseract.image_to_string(image, lang="tha+eng")
    for index in text_not_slip:
        transfer = re.findall(index, result)
        if transfer:
            predict = "false"
    for index in text_slip:
        transfer = re.findall(index, result)
        if transfer:
            predict = "true"
    if predict == "":
        predict = "false"
    return file, predict

def search_text(path, text_req):
    list_txt = []
    text_slip = []
    folder = os.listdir(path)
    texts = open("INS/texts/text_slip.txt", encoding="utf8")
    textss = texts.read()
    text_slip = textss.split(",")
    list_txt = text_req.split(",")
    texts.close()
    textn = open("INS/texts/text_not_slip.txt", encoding="utf8")
    textns = textn.read()
    text_not_slip = textns.split(",")
    textn.close()
    text_images = convert_img(path, folder, text_not_slip, text_slip)
    if list_txt != [""]:
        img_text_req = convert_img(path, folder, text_not_slip, list_txt)
    else:
        img_text_req = text_images
    print("search text succeed")
    return text_images, img_text_req