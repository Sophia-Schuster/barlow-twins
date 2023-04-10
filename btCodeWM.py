#Sophia Santonastasio Schuster 
#Barlow Twins and Siamese contrastive for Time Series in Python

#Used datasets:
# ECG 128 p/second  
# https://physionet.org/content/aftdb/
# https://physionet.org/content/shareedb/

######### IMPORTS, ENVIRONMENT SETTINGS ########

import os

import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  

import wfdb

import tsaug

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from copy import deepcopy

import random

from keras.models import load_model
import keras.losses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Initializing libiomp5md.dll, but found libiomp5 already initialized. Solved using:
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Consts
AUTO = tf.data.AUTOTUNE
CROP_TO = 32
SEED = 42
PROJECT_DIM = 2048
batch_size = 32 
epochs = 500

#Functions
def create_list():
    data = [] #save waves
    description = [] #save anotations

    file = 0

    pasta = os. getcwd()
    pasta = pasta + "\data"
    caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
    arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    dat = [arq for arq in arquivos if arq.lower().endswith(".dat")]

    files_name = []
    for texto in dat:
        files_name.append(texto.replace('.dat', ''))


    while file < len(files_name):
        data_name = files_name[file]

        record = wfdb.rdrecord(data_name, channels=[0]) 
        # Channels = 0 discart orange wave
        ann = wfdb.rdann(data_name, 'qrs', summarize_labels = 'true', return_label_elements = ['symbol']) 

        # print(ann.sample) # position of each annotation
        # print(ann.symbol) # simbol of each annotation

        # wfdb.show_ann_classes()
        # wfdb.show_ann_labels() #meaning of each simbol from description
        tam_janela = (ann.sample[0])  # defined by the first annotation position
        tam_janela = tam_janela//2

        y = 0

        for x in ann.sample:
            up = x+tam_janela
            down = x-tam_janela
            if (record.p_signal[down:up]).size > 0:
                data.append(record.p_signal[down:up])
                description.append(ann.symbol[y])
            y +=1

        file+=1

    #reducing description dimensionality
    description_uni = []
    for lista in description:
        for elemento in lista:
            description_uni.append(elemento)

    return description_uni, data

def reshape(features):
    # Two different sets of augmentations are applied to the same input 
    # sample X resulting in two distorted views of the same image (features_a, features_b).

    y = 0

    features_a = deepcopy(features)

    # Example no augmentation
    # print(features[30].shape)
    # plt.plot(features[30])
    # plt.title("original features[30]")
    # plt.show()

    while y<len(features):

        features_a[y].shape = features_a[y].shape[0]
        features_a[y] = np.array(features_a[y])

        # Change the temporal resolution of time series. 
        # The resized time series is obtained by linear interpolation of the original time series.
        features_a[y] = tsaug.Resize(size=224).augment(features_a[y],Y=None)

        y += 1
    return features_a
        
# Contrastive loss = mean( (1-true_value) * square(prediction) + true_value * square( max(margin-prediction, 0) ))
def loss(y_true, y_pred):
    margin = 1
    print(y_true)
    print(y_pred)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )

def cnn_block(input_data):
    x = Conv1D(16, (3), activation='tanh', padding='same')(input_data)
    x = MaxPooling1D((2), padding='same')(x)
    x = Conv1D(8, (3), activation='tanh', padding='same')(x)
    x = MaxPooling1D((2), padding='same')(x)
    x = Conv1D(1, (3), activation='tanh', padding='same')(x)
    encoded = MaxPooling1D((2), padding='same')(x)

    x = Conv1D(8, (3), activation='tanh', padding='same')(encoded)
    x = UpSampling1D((2))(x)
    x = Conv1D(8, (3), activation='tanh', padding='same')(x)
    x = UpSampling1D((2))(x)
    x = Conv1D(16, (3), activation='tanh', padding='same')(x)
    x = UpSampling1D((2))(x)
    decoded = Conv1D(1, (3), activation='tanh', padding='same')(x)

    # print("ENTROOOO")
    encoder = Model(input_data, encoded)
    encoder.summary()

    return decoded, encoder

def model(train_features):

    data_array = deepcopy(train_features)
    data_array = np.array(data_array)

    # adjusting position to be like: epochs, lenght, channels (example: (1276, 224, 1))

    serie_length = data_array.shape[1]
    print("Serie length:", serie_length)

    data_array.shape = data_array.shape[0], serie_length, 1 

    print("epochs, lenght, channels:", data_array.shape)

    #Create model

    serie_a_inp = Input((serie_length, 1), name='serie_a_inp')
    serie_b_inp = Input((serie_length, 1), name='serie_b_inp')

    feature_vector_A, encoder_model_A = cnn_block(serie_a_inp)
    feature_vector_B, encoder_model_B = cnn_block(serie_b_inp)

    concat = Concatenate()([feature_vector_A, feature_vector_B])

    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[serie_a_inp, serie_b_inp], outputs=output)

    return model, encoder_model_B

# def make_pairs(train_features_a,train_features_b):
#     aux = 0
#     x_pairs, label_pairs = [], []
#     while aux<(len(train_features_a)-2):

#         p = random.randint(0, 1)

#         if p==1:
#             label_pairs.append(1)
#             x_pairs.append([train_features_a[aux],train_features_b[aux]])
        
#         if p==0:
#             label_pairs.append(0)
#             x_pairs.append([train_features_a[aux],train_features_b[random.randint((aux+1),(len(train_features_a)-2))]])

#         aux +=1

#     return x_pairs, label_pairs


########## TIME SERIES WINDOWS ########
description_uni, data = create_list()

########## TRAINING AND VALIDATION SETS ########

test_features = data

print(f"Total test examples: {len(test_features)}")

########## AUGMENTATION ########
test_features_a= reshape(test_features)

########## MAKE PAIRS ########
test_pairs = test_features_a

test_pairs = np.array(test_pairs)

########## PREDICT ########
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# load the model from disk
modelo = load_model('encoder_model.h5', custom_objects={'loss':                   
loss}, compile = False)

predictions = modelo.predict(test_pairs)

print("Coded Data shape:", predictions.shape)
predictions.shape = predictions.shape[0], predictions.shape[1]


np.savetxt('encoder.txt', predictions)
np.savetxt('description.txt',description_uni, fmt='%s')
np.savetxt('data_original.txt', test_pairs)

