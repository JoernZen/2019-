import os
import glob
import random
import keras
import json
import numpy as np
import pandas as pd
import resnet
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.callbacks import *
from keras import *
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras import backend as K
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_shape=(40,120,3)
# input image dimensions
img_rows, img_cols = 40,120
# The CIFAR10 images are RGB.
img_channels = 3

def compare(predict,label):
    n=predict[0].shape[0]
    label_argmax=np.array(label).argmax(axis=2)
    predict_argmax=np.array(predict).argmax(axis=2)
    return (label_argmax==predict_argmax).all(axis=(0,)).sum()/n

def int_to_tensor(i,n):
    t=np.zeros((n,))
    t[i]=1.0
    return t

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
        
# 加载训练图片和验证码文字
def load_train_picture_and_label(config):

    train_label_c0=[]
    train_label_c1=[]
    train_label_c2=[]
    train_label_c3=[]

    train_images = []

    image_dir=config['image_source']

    random.seed()
    pd_data = pd.read_csv(image_dir+'train_label.csv')
    for i in range(0, pd_data.shape[0]):
    #for i in range(0,500):
        file = pd_data.iloc[i, 0]
        label = pd_data.iloc[i, 1]
        image_data = image.load_img(image_dir+file)
        image_tensor = image.img_to_array(image_data)
        label_tensor_0=int_to_tensor(base64_char_to_int(label[0]),62)
        label_tensor_1=int_to_tensor(base64_char_to_int(label[1]),62)
        label_tensor_2=int_to_tensor(base64_char_to_int(label[2]),62)
        label_tensor_3=int_to_tensor(base64_char_to_int(label[3]),62)
        train_images.append(image_tensor)
        train_label_c0.append(label_tensor_0)
        train_label_c1.append(label_tensor_1)
        train_label_c2.append(label_tensor_2)
        train_label_c3.append(label_tensor_3)

    train_images=np.array(train_images,dtype='float')/255.0

    train_label_c0=np.array(train_label_c0,dtype='float')
    train_label_c1=np.array(train_label_c1,dtype='float')
    train_label_c2=np.array(train_label_c2,dtype='float')
    train_label_c3=np.array(train_label_c3,dtype='float')

    train_labels = [train_label_c0,train_label_c1,train_label_c2,train_label_c3]
    return train_images, train_labels

# 加载测试图片和文件名
def load_predict_pictrue(config):
    image_name_list = []
    image_data_list = []
    image_dir=config['image_source']
    for i in range(1,5001):
        file=str(i)+'.jpg'
        image_name_list.append(file)
        image_data = image.load_img(image_dir+file)
        image_tensor = image.img_to_array(image_data)
        image_data_list.append(image_tensor)
    image_data_list=np.array(image_data_list,dtype='float')/255.0
    return image_data_list, image_name_list

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def main():
    config=json.loads(open('./config.json',encoding='utf-8').read())
    if config['Model']['give_up']:
        model=creat_model(image_shape)
    else:
        if os.path.exists(config['Model']['file_path']):
            model=load_model(config['Model'])
        else:
            model=creat_model(image_shape)
    if config['Train']['enable']:
        train_config=config['Train']
        train_images, train_labels=load_train_picture_and_label(train_config['Train_set'])
        if train_config['cycle_train']:
            while True:
                train(model,train_images, train_labels,train_config)
                store_model(model,config['Model'])
        else:
            train(model,train_images, train_labels,train_config)
            store_model(model,config['Model'])
    if config['Test']['enable']:
        predict_config=config['Test']
        predict_image,predict_file_name=load_predict_pictrue(predict_config['Test_set'])
        predict_label=predict(model,predict_image)
        save_predict_result(predict_file_name,predict_label,predict_config)

def creat_model(input_shape):
    model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), 62)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def train(model,train_images, train_labels,config):
    callbacks = [
        EarlyStopping(patience=100, verbose=1, mode='auto')
    ]
    if config['show_graphics']:
        callbacks_history=LossHistory()
        callbacks.append(callbacks_history)
    
    if config['expansion']:
        
        def expansion_train_images(images,labels):
            i=0
            c=50
            while True:
                matrix=np.random.rand(3,3)/3
                for r in range(0,100):
                    image=images[i:i+c]
                    image=np.matmul(image,matrix)
                    yield (image,[labels[0][i:i+c],labels[1][i:i+c],labels[2][i:i+c],labels[3][i:i+c]])
                    i+=c
                    if i>=4500:
                        i=0
        def expansion_test_images(images,labels):
            i=4500
            c=50
            while True:
                for r in range(0,100):
                    image=images[i:i+c]
                    yield (image,[labels[0][i:i+c],labels[1][i:i+c],labels[2][i:i+c],labels[3][i:i+c]])
                    i+=c
                    if i>=5000:
                        i=4500
        model.fit_generator(
              expansion_train_images(train_images,train_labels),
              steps_per_epoch=800,
              epochs=config['epoch'],
              #batch_size=config['batch_size'],
              validation_data=expansion_test_images(train_images,train_labels),
              validation_steps=5,
              verbose=config['display_mode'],
              callbacks=callbacks,
        )
        #print(compare(model.predict(np.matmul(train_images[-500:],np.random.rand(3,3)/3)),[p[-500:] for p in train_labels]))
        print(compare(model.predict(train_images[:-500]),[p[:-500] for p in train_labels]))
        print(compare(model.predict(train_images[-500:]),[p[-500:] for p in train_labels]))
    else:
#        plot(model, to_file="model.png", show_shapes=True)
#        Image('model.png')
        model.fit(train_images,train_labels,
              epochs=config['epoch'],
              batch_size=config['batch_size'],
              verbose=config['display_mode'],
              callbacks=callbacks,
              validation_split=0.1
              )
        print(compare(model.predict(train_images[:-500]),[p[:-500] for p in train_labels]))
        print(compare(model.predict(train_images[-500:]),[p[-500:] for p in train_labels]))
    if config['show_graphics']:
        callbacks_history.loss_plot('epoch')

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

def save_predict_result(file_names,labels,config):
    frame = pd.DataFrame({'ID':file_names,'label':labels},columns=['ID','label'])
    frame.to_csv(config['test_result_file'])

def load_model(config):
    return keras.models.load_model(config['file_path'])


def store_model(model,config):
    model.save(config['file_path'])

if __name__ == "__main__":
    main()
