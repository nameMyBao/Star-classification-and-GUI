import pandas as pd
import re
import string
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras_tuner as kt

from tkinter import ttk


def data_in(direct):
    
    feature = pd.read_csv(direct,usecols=['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 
                                          'Absolute magnitude (Mv)','Star color', 'Spectral Class'])
    
    feature = feature.to_numpy()
    label = pd.read_csv(direct, usecols=['Star type'])
    label = label.to_numpy()
    label = np.asarray(label).astype('float32')
  
    
    feature[:, 4]=custom_input(feature[:, 4])
    feature[:, 5]=custom_input(feature[:, 5])
    
   
    
    # unique_color =np.unique(feature[:, 4], return_inverse=True)[0]
    # unique_spectral = np.unique(feature[:, 5], return_inverse=True)[0]
    # unique_spectral2 = np.unique(feature[:, 5], return_inverse=True)[1]
    # unique_spectralref = np.unique(unique_spectral2, return_inverse=True)[0]
    # unique_color2 = np.unique(feature[:, 4], return_inverse=True)[1]
    # unique_colorref=np.unique(unique_color2, return_inverse=True)[0]
   
   
    

    feature[:, 4] = np.unique(feature[:, 4], return_inverse=True)[1].tolist()
    feature[:, 5] = np.unique(feature[:, 5], return_inverse=True)[1].tolist()
   
  
    

   

    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    mean = scaler.mean_
    scale = scaler.scale_
    print("mean:",mean)
    print("scale:",scale)
   
    

    train_feature, test_feature, train_label, test_label = train_test_split(feature,
                                                                            label, test_size=0.2,
                                                                            random_state=42)
    
    
    # print("train feature",train_feature)
    # print("test feature",test_feature)
    return train_feature, test_feature, train_label, test_label, label,mean,scale

def build_model ():
    
    model = tf.keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(6,)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(200,activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(200,activation='sigmoid',))
    model.add(keras.layers.Dense(400,activation='sigmoid',))
    # model.add(keras.layers.Dropout(rate=0.2))
    # model.add(keras.layers.Dense(400,activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    # model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(6,activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate= 0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model

def train_model (model, feature, label, test_feature, test_label):
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=17, restore_best_weights=True)
    history = model.fit(feature, label, epochs=100, batch_size=64, 
                        validation_data=(test_feature, test_label),callbacks=[early_stopping])
    trained_epochs = history.epoch
    history_df = pd.DataFrame(history.history)
    accuracy = history_df['accuracy']
    loss = history_df['loss']
    val_acc = history_df['val_accuracy']
    val_loss = history_df['val_loss']
    
    return trained_epochs, accuracy, loss, val_acc, val_loss



def plot_loss_curve(epochs,  accuracy, loss, val_acc, val_loss):
    
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(epochs, accuracy, label='accuracy',color='orange')
    plt.plot(epochs, val_acc, label='validation accuracy',color='red')
    plt.legend()
    plt.show()
    plt.plot(epochs, loss, label='loss',color='blue')
    plt.plot(epochs, val_loss, label='validation loss',color='brown')
    plt.legend()
    plt.show()
    
def custom_input(input):
    
  lowercase = tf.strings.lower(input)
  newstring1 = tf.strings.regex_replace(lowercase, ' ', '')
  newstring = tf.strings.regex_replace(newstring1, '-', '')
  return newstring



def custom_data(data,mean,scale):
    i=0
    for x in data:
        data[i]=((data[i])-mean)/scale
        i+=1
    return data
    
# star choice probability
def star_percent(root,percent):
    # Brown Dwarf
    Bd_value_label = ttk.Label(
        root,
        text='Brown Dwarf (%):'
    )
    Bd_value_label.grid(
        column=0,
        row=19,
        sticky='we'
    )
    Bdval_label = ttk.Label(
        root,
        text=round(percent[0],4)
    )
    Bdval_label.grid(
        column=0,
        row=20,
        columnspan=1,
        sticky='w'
    )


    # Red Dwarf
    Rd_value_label = ttk.Label(
        root,
        text='Red Dwarf (%):'
    )
    Rd_value_label.grid(
        column=1,
        row=19,
        sticky='we'
    )
    Rdval_label = ttk.Label(
        root,
        text=round(percent[1],4)
    )
    Rdval_label.grid(
        column=1,
        row=20,
        columnspan=1,
        sticky='w'
    )

    # White Dwarf
    Wd_value_label = ttk.Label(
        root,
        text='White Dwarf (%):'
    )
    Wd_value_label.grid(
        column=2,
        row=19,
        sticky='we'
    )
    Wdval_label = ttk.Label(
        root,
        text=round(percent[2],4)
    )
    Wdval_label.grid(
        column=2,
        row=20,
        columnspan=1,
        sticky='n'
    )

    # Main Sequence
    Ms_value_label = ttk.Label(
        root,
        text='Main Sequence (%):'
    )
    Ms_value_label.grid(
        column=0,
        row=21,
        sticky='we'
    )
    Msval_label = ttk.Label(
        root,
        text=round(percent[3],4)
    )
    Msval_label.grid(
        column=0,
        row=22,
        columnspan=1,
        sticky='w'
    )

    # Super Giant
    Sg_value_label = ttk.Label(
        root,
        text='Super Giant (%):'
    )
    Sg_value_label.grid(
        column=1,
        row=21,
        sticky='we'
    )
    Sgval_label = ttk.Label(
        root,
        text=round(percent[4],4)
    )
    Sgval_label.grid(
        column=1,
        row=22,
        columnspan=1,
        sticky='w'
    )

    # Hypergiant
    Hp_value_label = ttk.Label(
        root,
        text='Hypergiant (%):'
    )
    Hp_value_label.grid(
        column=2,
        row=21,
        sticky='we'
    )
    Hpval_label = ttk.Label(
        root,
        text=round(percent[5],4)
    )
    Hpval_label.grid(
        column=2,
        row=22,
        columnspan=1,
        sticky='n'
    )

