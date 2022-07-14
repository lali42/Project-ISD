import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"INS\libary\TesseractOCR\tesseract.exe"

img = cv2.imread('D:/2022/Project/INS/DATASET/Train/not_slip/IMG_0143.JPG')
custom_config = r'-l tha+eng'
print(pytesseract.image_to_string(img, config=custom_config))

data = pytesseract.image_to_data(
    img, config=custom_config, output_type=Output.DICT)
keys = list(data.keys())
totalBox = len(data['text'])
for i in range(totalBox):
    (x, y, w, h) = (data['left'][i], data['top']
                    [i], data['width'][i], data['height'][i])
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
print(''.join(data['text']))
