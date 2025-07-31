# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:24:29 2022

@author: Dell
"""

import pickle
from keras.models import model_from_json

# load json and create model
json_file = open(r'C:\Users\Dell\Downloads\Base_model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"C:\Users\Dell\Downloads\Base_model_1.h5")

from keras.models import Model
import numpy as np
from numpy import load
layer_name='my_dense'
intermediate_layer_model11 = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.get_layer(layer_name).output)

import pickle
xgboost=pickle.load(open('new_xgboost.sav','rb'))

#def pred(prediction_1):
    #if prediction_1==0:
     #   print('Mild dementia')
    #elif prediction_1==1:
     #   print('Moderate Dementia')
    #elif prediction_1==2:
     #   print('Non Dementia')
    #else:
     #   print('very mild Dementia')

import cv2


#mg=r'C:\Users\Dell\Desktop\predict data\Non Dementia.jpg'
#mg_1=cv2.imread(mg)
#mg_1=cv2.resize(mg_1,(224,224))
#input_img2=np.expand_dims(mg_1,axis=0)
#intermediate_test2 = intermediate_layer_model11.predict(input_img2)
#intermediate_test_feat2=intermediate_test2.reshape(intermediate_test2.shape[0],-1)
#prediction_2=xgboost.predict(intermediate_test_feat2)[0]
#pred(prediction_2)

import streamlit as st
from PIL import Image
from skimage.transform import resize

def pred(prediction_1):
    if prediction_1==0:
        st.write('Mild dementia')
    elif prediction_1==1:
        st.write('Moderate Dementia')
    elif prediction_1==2:
        st.write('Non Dementia')
    else:
        st.write('very mild Dementia')
st.title('Alzheimers Disease prediction')
upload_file=st.file_uploader('choose a image',type='jpg')
if upload_file is not None:
    image=Image.open(upload_file)
    st.image(image,caption='upload image')
    
    if st.button('PREDICT'):
        image_2=Image.open(upload_file)
        mg_1=np.array(image_2)
        mg_1=resize(mg_1,(224,224,3))
        input_img2=np.expand_dims(mg_1,axis=0)
        intermediate_test2 = intermediate_layer_model11.predict(input_img2)
        intermediate_test_feat2=intermediate_test2.reshape(intermediate_test2.shape[0],-1)
        prediction_3=xgboost.predict(intermediate_test_feat2)[0]
        pred(prediction_3) 








