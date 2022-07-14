import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"INS\libary\TesseractOCR\tesseract.exe"

list_color = [
    (np.array([0, 0, 0]), np.array([160, 160, 160])),
    (
        np.array([0, 0, 0]),
        np.array([170, 170, 170]),
    ),
]
image = cv2.imread("Image/slip43.JPG")
# image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA)
image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
for low, high in list_color:
    mask = cv2.inRange(image, low, high)
    image = cv2.bitwise_and(image, image, mask=mask)
    txt = pytesseract.image_to_string(image, lang="tha+eng")
    print(txt)
    print("==============")
    cvt_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    imgplot = plt.imshow(cvt_image)
    plt.show()