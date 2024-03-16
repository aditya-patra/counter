import io

from flask import Flask, render_template, request, jsonify
import cv2
import base64
from PIL import Image
import imageio.v2 as imageio
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
# import scipy
import tensorflow as tf
import webbrowser
import numpy as np

app = Flask(__name__, template_folder='templates')


@app.route('/')
def try2():
    return render_template('try2.html')


@app.route('/get_text', methods=['POST'])
def get_text():
    name = request.get_json()
    print(name)
    return name


@app.route('/final_page', methods=['POST'])
def final_page():
    print("im here")
    form_data = request.get_json()
    print(str(form_data))
    #print(form_data)
    # read the image
    img = base64.b64decode(form_data.split(',')[1])
    #print(type(img))
    img = Image.open(io.BytesIO(img))
    #print(type(img))
    # img = imageio.imread(img)
    img = np.array(img, dtype=np.uint8)
    #print(type(img))
    totalAmt = 0
    rimg = img.copy()
    cimg = img.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # make a copy of the original image
    imgdiv = []
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply a blur using the median filter
    img = cv2.medianBlur(img, 5)
    # finds the circles in the grayscale image using the Hough transform
    try:
        circles1 = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9,
                                   minDist=80, param1=110, param2=60, minRadius=1, maxRadius=3000)
        #print(type(circles))
        imgName = './CreatedImages/5.jpg'
        #cv2.imwrite(imgName, img)
        circles = np.round(circles1[0, :]).astype("int")
        for (x, y, r) in circles:
            try:
                imgdiv.append(rimg[y - r - 5:y + r + 5, x - r - 5:x + r + 5])
                # draw the outer circle in green
                cv2.circle(cimg, (x, y), r, (0, 255, 0), 2)
                # draw the center of the circle in red
                cv2.circle(cimg, (x, y), 2, (0, 0, 255), 3)
            except:
                pass
                #alert("Img not working")

        # print the number of circles detected
        # print("Number of circles detected:", len(circles))
        # print(len(imgdiv))
        # save the image, convert to BGR to save with proper colors
        cv2.imwrite("coins_circles_detected.png", cimg)
        # show the image
        # plt.imshow(cimg)
        # plt.show()
        f = 1
        model = tf.keras.models.load_model("checkpoint_path")
        #cubic = 3
        #return str(cubic)
        #'''
        # dir_path = "C:/Users/aadip/PycharmProjects/Shape_Classifier/CreatedImages"
        for e in imgdiv:
            print(e)
            #imgName = './CreatedImages/' + str(f) + '.jpg'
            #cv2.imwrite(imgName, e)
            # plt.imshow(e)
            # plt.show()
            PILImg = Image.fromarray(e)
            #print('d')
            #print(type(PILImg))
            #PILImg.show()
            PILImg = PILImg.resize((75, 75))
            X1 = tf.keras.preprocessing.image.img_to_array(PILImg)
            X = np.expand_dims(X1, axis=0)
            images = np.vstack([X])
            val = model.predict(images)
            print(val)
            if str(val) == "[[1. 0. 0. 0.]]":
                totalAmt += 10
            elif str(val) == "[[0. 1. 0. 0.]]":
                totalAmt += 5
            elif str(val) == "[[0. 0. 1. 0.]]":
                totalAmt += 1
            elif str(val) == "[[0. 0. 0. 1.]]":
                totalAmt += 25
            else:
                print(val)
            f += 1
    except:
        totalAmt = 0
    return str(totalAmt)
    #'''


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5001,
        debug=True
    )
