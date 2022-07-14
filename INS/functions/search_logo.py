from unittest import result
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import join

images_path = "Image/slip"
templates_path = "INS/DATASET/LOGO/"


def search_logo(images_path, templates_path):
    print(images_path, templates_path)
    color = (0, 0, 255)
    threshold = 0.8
    search_result = ""
    images_predict = {}
    template_size = (89, 89)

    # list_image = [
    #     images_path + f for f in listdir(images_path) if listdir(join(images_path, f))
    # ]

    list_template = [
        templates_path + f
        for f in listdir(templates_path)
        if listdir(join(templates_path, f))
    ]

    ldseg = np.array(listdir(templates_path))
    count_ldseg = 0
    count = 0

    # print(list_template)
    # Image
    # for images in list_image:
    for img in listdir(images_path):
        count += 1
        # print(count)
        # Templates
        for templates in list_template:
            count_ldseg += 1
            if count_ldseg == 12:
                count_ldseg = 0

            for temp in listdir(templates):
                file_temp = join(templates, temp)

                file_img = join(images_path, img)

                img_class = ldseg[count_ldseg]

                img_rgb = cv2.imread(file_img)
                x, y = 0, 0
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                template = cv2.imread(file_temp, 0)
                # img_rgb = img_rgb[y:h/2, x:x+w]

                # if w == h and (h, w) > (137, 137):
                #     template_resize = cv2.resize(
                #         template, template_size, interpolation=cv2.INTER_AREA
                #     )
                # else:

                template_resize = template

                # w, h = template.shape[::-1]

                w_img = img_rgb.shape[0]
                h_img = img_rgb.shape[1]
                w_temp = template_resize.shape[0]
                h_temp = template_resize.shape[1]

                if w_img <= w_temp or h_img <= h_temp:
                    search_result = "false"
                else:
                    res = cv2.matchTemplate(
                        img_gray, template_resize, cv2.TM_CCOEFF_NORMED
                    )
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # top_left = max_loc
                    # bottom_right = (top_left[0] + w, top_left[1] + h)

                    # cv2.rectangle(img_rgb, top_left, bottom_right, color, 2)
                    # print(top_left, bottom_right)

                    if max_val >= threshold:
                        search_result = img_class
                        # print(img)
                        # print(min_val, max_val)
                        # print(search_result)
                        # plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
                        # plt.title(img + "result : " + search_result)
                        # plt.show()
                        if search_result == "Not_Slip":
                            search_result = "false"
                            images_predict[img] = [search_result, max_val]
                            break
                        else:
                            search_result = "true"

                    else:
                        search_result = "false"

                if img in images_predict:
                    if max_val > images_predict[img][1]:
                        images_predict[img] = [search_result, max_val]
                else:
                    images_predict[img] = [search_result, max_val]

    return images_predict

    # # loc = np.where( res >= threshold)
    # # for pt in zip(*loc[::-1]):
    # #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), color, 2)
    # #     count += 1


# images_predict = search_logo(images_path, templates_path)
# print(images_predict)
