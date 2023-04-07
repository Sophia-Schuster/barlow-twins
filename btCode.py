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

        for x in ann.sample:
            up = x+tam_janela
            down = x-tam_janela
            if (record.p_signal[down:up]).size > 0:
                data.append(record.p_signal[down:up])
        
        description.append(ann.symbol)

        file+=1

    #reducing description dimensionality
    description_uni = []
    for lista in description:
        for elemento in lista:
            description_uni.append(elemento)
    
    return description_uni, data

def augmentation(features):
    # Two different sets of augmentations are applied to the same input 
    # sample X resulting in two distorted views of the same image (features_a, features_b).

    y = 0

    features_a = deepcopy(features)
    features_b = deepcopy(features)

    # Example no augmentation
    # print(features[30].shape)
    # plt.plot(features[30])
    # plt.title("original features[30]")
    # plt.show()

    while y<len(features):

        features_a[y].shape = features_a[y].shape[0]
        features_b[y].shape = features_b[y].shape[0]
        features_a[y] = np.array(features_a[y])
        features_b[y] = np.array(features_b[y])

        # Change the temporal resolution of time series. 
        # The resized time series is obtained by linear interpolation of the original time series.
        features_a[y] = tsaug.Resize(size=224).augment(features_a[y],Y=None)
        features_b[y] = tsaug.Resize(size=224).augment(features_b[y],Y=None)

        # Gaussian blurring substituted for TIMEWARP        
        # The augmenter drifts the value of time series from its original values randomly and smoothly. 
        # The extent of drifting is controlled by the maximal drift and the number of drift points.
        features_a[y] = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(features_a[y],Y=None)

        # random aug for a 
        # Horizontal flipping applied with probability p=0.5 substituted for REVERSE 
        features_a[y] = tsaug.Reverse(prob=0.5).augment(features_a[y],Y=None)
        # Color-jittering applied with probability p=0.8 . substituted for ADDNOISE 
        # Add random noise to time series.
        features_a[y] = tsaug.AddNoise(scale=0.01,prob=0.8).augment(features_a[y],Y=None)
        # Conversion to grayscale with probability p=0.2 substituted for POOL 
        # Reduce the temporal resolution without changing the length.
        features_a[y] = tsaug.Pool(size=2,prob=0.2).augment(features_a[y],Y=None)
        
        # Same example with augmentation
        # if (y == 30):
        #     aux = features_a[30]
        #     plt.plot(features_a[30])
        #     plt.title("serie a[30] pós aug")
        #     plt.show()

        # random aug for b 
        # Horizontal flipping applied with probability p=0.5 substituted for REVERSE 
        features_b[y] = tsaug.Reverse(prob=0.5).augment(features_b[y],Y=None)
        # Color-jittering applied with probability p=0.8 . substituted for ADDNOISE 
        # Add random noise to time series.
        features_b[y] = tsaug.AddNoise(scale=0.01,prob=0.8).augment(features_b[y],Y=None)
        # Conversion to grayscale with probability p=0.2 substituted for POOL 
        # Reduce the temporal resolution without changing the length.
        features_b[y] = tsaug.Pool(size=2,prob=0.2).augment(features_b[y],Y=None)
        # Gaussian blurring (time warp) applied with probability p=0.1 .
        features_b[y] = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3, prob=0.1).augment(features_b[y],Y=None)
        # Solarization (max_value — pixel + min_value) applied with probability p=0.2 .
        # substituted for Convolve, Convolve time series with a kernel window.
        features_b[y] = tsaug.Convolve(window="flattop", size=11,prob=0.1).augment(features_b[y],Y=None)

        y += 1
    return features_a, features_b
        
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

    print("ENTROOOO")
    encoder = Model(input_data, encoded)
    encoder.summary()
    encoder.save('encoder_{}.h5'.format("ecg_database"))

    return decoded

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

    feature_vector_A = cnn_block(serie_a_inp)
    feature_vector_B = cnn_block(serie_b_inp)

    concat = Concatenate()([feature_vector_A, feature_vector_B])

    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[serie_a_inp, serie_b_inp], outputs=output)

    return model

def make_pairs(train_features_a,train_features_b):
    aux = 0
    x_pairs, label_pairs = [], []
    while aux<(len(train_features_a)-2):

        p = random.randint(0, 1)

        if p==1:
            label_pairs.append(1)
            x_pairs.append([train_features_a[aux],train_features_b[aux]])
        
        if p==0:
            label_pairs.append(0)
            x_pairs.append([train_features_a[aux],train_features_b[random.randint((aux+1),(len(train_features_a)-2))]])

        aux +=1

    return x_pairs, label_pairs

def print_pair(pairs, labels):
    v = random.randint(0, (len(pairs)-1))
    plt.subplot(1,2,1)
    plt.plot(pairs[v,0])
    plt.title("Exemplo de par com augmentation")
    plt.subplot(1,2,2)
    plt.plot(pairs[v,1])
    if labels[v] > 0.5:
        plt.title("True Pair")
    else:
        plt.title("False Pair")
    plt.show()
    return v

def plt_metric(history, metric, title, has_valid=True):
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history[metric])
        plt.legend(["train"])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


########## TIME SERIES WINDOWS ########
description_uni, data = create_list()

########## TRAINING AND VALIDATION SETS ########
half = (len(data)-1)//2
up = len(data)-1

train_features = data[:half]  
test_features = data[(half+1):up] 
train_labels = description_uni[:half]
test_labels = description_uni[(half+1):up]

print(f"Total training examples: {len(train_features)}")
print(f"Total test examples: {len(test_features)}")

########## AUGMENTATION ########
train_features_a, train_features_b = augmentation(train_features)
test_features_a, test_features_b = augmentation(test_features)

########## CREATE MODEL ########
modelo = model(train_features_a)

# load the model from disk
# modelo = load_model('model.h5', custom_objects={'loss':                   
# loss})

########## MAKE PAIRS ########
train_pairs, label_pairs = make_pairs(train_features_a,train_features_b)
test_pairs, label_test_pairs = make_pairs(test_features_a,test_features_b)

train_pairs = np.array(train_pairs)
test_pairs = np.array(test_pairs)
label_pairs = np.array(label_pairs)
label_test_pairs = np.array(label_test_pairs)

label_pairs.shape = label_pairs.shape[0]
label_test_pairs.shape = label_test_pairs.shape[0]

#example, 0 is false pair
v = print_pair(train_pairs, label_pairs)


########## COMPILE WITH CONSTRASTIVE LOSS  ########
modelo.compile(optimizer='RMSprop', loss=loss, metrics=['accuracy'])
modelo.summary()
#loss mse good too

########## TRAIN  ########
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

label_pairs = np.asarray(label_pairs).astype('float32').reshape((-1,1))
label_test_pairs = np.asarray(label_test_pairs).astype('float32').reshape((-1,1))

# Keep 50% of train_val  in validation set
# half = (len(train_pairs)-1)//2
# up = len(train_pairs)-1
# history = modelo.fit([train_pairs[:half, 0, :], train_pairs[:half, 1, :]],
#           label_pairs[:half],
#           validation_data=([train_pairs[(half+1):up, 0, :], 
#                             train_pairs[(half+1):up, 1, :]],
#                            label_pairs[(half+1):up]),
#           epochs=epochs,
#           batch_size=batch_size,
#           callbacks=[callback])

#without validation data
history = modelo.fit([train_pairs[:, 0, :], train_pairs[:, 1, :]],
          label_pairs,
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[callback])

# Save the trained model
# modelo.save('model.h5')

########## EVALUATE ########
# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

#evaluate the model
results = modelo.evaluate([test_pairs[:, 0, :], test_pairs[:, 1, :]], label_test_pairs)
print("test loss, test acc:", results)

#Visualize the predictions
predictions = modelo.predict([test_pairs[:, 0, :], test_pairs[:, 1, :]])
print(f"Prediction: {predictions[v,0]}")
print(f"Prediction: {predictions[v,1]}")
plt.subplot(1,2,1)
plt.plot(test_pairs[v,0])
if predictions[v,0] > 0.5:
    plt.title("True Pair")
else:
    plt.title("False Pair")
plt.subplot(1,2,2)
plt.plot(test_pairs[v,1])
if predictions[v,1] > 0.5:
    plt.title("True Pair")
else:
    plt.title("False Pair")
plt.show()
