from platform import win32_edition
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784",version = 1,return_X_y=True)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500)
xtrain_scaled = xtrain/255.0
xtestscaled = xtest/255.0

classify = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xtrain_scaled,ytrain)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert("L")
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized - min_pixel,0,255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_prediction = classify.predict(test_sample)
    return test_prediction[0]