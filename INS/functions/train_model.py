import tensorflow as tf
import numpy as np
import numpy
import cv2
import os
from tqdm import tqdm
import keras
from os import listdir
from os.path import join

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import seaborn as sns

width = 128
num_classes = 2


def img2data(path):
    rawImgs = []
    labels = []
    count = 0
    for imagePath in path:
        for item in tqdm(os.listdir(imagePath)):
            count += 1
            file = os.path.join(imagePath, item)
            l = imagePath.split("\\")[-1]
            if l == "slip":
                labels.append([1, 0])
            elif l == "not_slip":
                labels.append([0, 1])
            img = cv2.imread(file, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, width))
            rawImgs.append(img)
    return count, rawImgs, labels


def train(trainpath, testpath):
    trainImg = [
        trainpath + f for f in listdir(trainpath) if listdir(join(trainpath, f))
    ]
    testImg = [testpath + f for f in listdir(testpath) if listdir(join(testpath, f))]

    total_train, x_train, y_train = img2data(trainImg)
    total_test, x_test, y_test = img2data(testImg)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                input_shape=(width, width, 3),
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(
                128,
                activation="relu",
            ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        loss="BinaryCrossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    batch_size = 32
    epochs = 25

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )

    # Create figure with secondary y-axis
    fig_loss = make_subplots(specs=[[{"secondary_y": True}]])
    fig_acc = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pre = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rec = make_subplots(specs=[[{"secondary_y": True}]])

    # create loss image
    fig_loss.add_trace(
        go.Scatter(y=history.history["val_loss"], name="val loss"),
        secondary_y=False,
    )
    fig_loss.add_trace(
        go.Scatter(y=history.history["loss"], name="loss"),
        secondary_y=False,
    )
    fig_loss.update_layout(title_text="Loss form Model")
    fig_loss.update_xaxes(title_text="Epoch")
    fig_loss.update_yaxes(title_text="<b>Loss</b>", secondary_y=False)

    # create accuracy image
    fig_acc.add_trace(
        go.Scatter(y=history.history["val_accuracy"], name="val accuracy"),
        secondary_y=True,
    )
    fig_acc.add_trace(
        go.Scatter(y=history.history["accuracy"], name="accuracy"),
        secondary_y=True,
    )
    fig_acc.update_layout(title_text="Accuracy form Model")
    fig_acc.update_xaxes(title_text="Epoch")
    fig_acc.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=True)

    # create precision image
    fig_pre.add_trace(
        go.Scatter(y=history.history["val_precision"], name="val precision"),
        secondary_y=True,
    )
    fig_pre.add_trace(
        go.Scatter(y=history.history["precision"], name="precision"),
        secondary_y=True,
    )
    fig_pre.update_layout(title_text="Precision form Model")
    fig_pre.update_xaxes(title_text="Epoch")
    fig_pre.update_yaxes(title_text="<b>Precision</b>", secondary_y=True)

    # create recall image
    fig_rec.add_trace(
        go.Scatter(y=history.history["val_recall"], name="val recall"),
        secondary_y=True,
    )
    fig_rec.add_trace(
        go.Scatter(y=history.history["recall"], name="recall"),
        secondary_y=True,
    )
    fig_rec.update_layout(title_text="Recall form Model")
    fig_rec.update_xaxes(title_text="Epoch")
    fig_rec.update_yaxes(title_text="<b>Recall</b>", secondary_y=True)

    # y_prediction = model.predict(x_test)
    # zcm = tf.math.confusion_matrix(y_test, y_prediction)
    # print("group_zcm", zcm)
    
    # ax = sns.heatmap(zcm, annot=True, cmap='Blues')
    # ax.set_title('Confusion Matrix\n\n')
    # ax.set_xlabel('\nPredicted Values')
    # ax.set_ylabel('Actual Values ')
    # ax.xaxis.set_ticklabels(['True','False'])
    # ax.yaxis.set_ticklabels(['True','False'])
    # plt.show()

    if not os.path.exists("Image/plot_fig"):
        os.mkdir("Image/plot_fig")
    else:
        fig_loss.write_image("Image/plot_fig/loss.png")
        fig_acc.write_image("Image/plot_fig/accuracy.png")
        fig_pre.write_image("Image/plot_fig/precision.png")
        fig_rec.write_image("Image/plot_fig/recall.png")
    test_loss, test_acc, precision, recall = model.evaluate(x_test, y_test, verbose=2)
    print(test_loss, test_acc, precision, recall)
    return model, test_acc, test_loss, precision, recall, total_train, total_test
