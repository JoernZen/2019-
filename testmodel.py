# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""

import numpy as np
import pandas as pd
import keras
from keras.preprocessing import image


def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    pass
    model=keras.models.load_model("./model/model.h5")
    predict_image,predict_file_name=load_predict_pictrue(testpath)
    predict_label=predict(model,predict_image)
    # the format of result-file
    # 这里可以生成结果文件
    ids = [str(x) + ".jpg" for x in range(1, 5001)]
    labels = predict_label
    df = pd.DataFrame([ids, labels]).T
    df.columns = ['ID', 'label']
    return df
    
def load_predict_pictrue(testpath):
    image_name_list = []
    image_data_list = []
    image_dir=testpath
    for i in range(1,5001):
        file=str(i)+'.jpg'
        image_name_list.append(file)
        image_data = image.load_img(image_dir+file)
        image_tensor = image.img_to_array(image_data)
        image_data_list.append(image_tensor)
    image_data_list=np.array(image_data_list,dtype='float')/255.0
    return image_data_list, image_name_list

def predict(model,images):
    predict_tensor=model.predict(images)
    label_id_tensor=np.array(predict_tensor).argmax(axis=2)
    label=[]
    for j in range(0,label_id_tensor.shape[1]):
        label.append('')
        for i in range(0,4):
            label_number=label_id_tensor[i,j]
            label[-1]+=(int_to_base64_char(label_number))
    return label

def int_to_base64_char(i):
    if 0<=i<=9:
        return chr(i+48)
    if 10<=i<=35:
        return chr(i+55)
    if 36<=i<=61:
        return chr(i+61)

def base64_char_to_int(char):
    c=ord(char)
    if 48<=c<=57:
        return c-48
    if 65<=c<=90:
        return c-65+10
    if 97<=c<=122:
        return c-97+36