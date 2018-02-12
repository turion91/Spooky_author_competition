# -*- coding: utf-8 -*-
"""Contains an LSTM wrapper for the book author competition in Kaggle.
"""

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '0.0.1'
__date__ = '2018-02-06'

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers import Embedding
from keras.layers import LSTM
from keras.regularizers import l2
import matplotlib.pyplot as plt
import keras
le = preprocessing.LabelEncoder()

def lstm(DATA_DIR, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, 
         output_dim, units, kernel_regularizer, 
         activity_regularizer, dropout, classes, activation_function, loss, 
         learning_rate, metric, batch_size, epochs, *args, **kwargs):
    '''
    Function used to create the LSTM model
    
    '''
    train = pd.read_csv(DATA_DIR)
    len(train)
    texts = train['text']
    labels = (train['author'])
    labels = le.fit_transform(labels)
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  
        
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
    one_hot_labels = keras.utils.to_categorical(labels, num_classes = classes)
        
    x_train, x_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.33, random_state=14)        
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, output_dim))
    model.add(LSTM(units = units, kernel_regularizer = kernel_regularizer, activity_regularizer = activity_regularizer))
    model.add(Dropout(rate = dropout))
    model.add(Dense(units = classes, activation = activation_function))
    
    optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    
    model.compile(loss = loss,
                  optimizer = optimizer,
                  metrics = metric)
        
        
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_test,y_test))
    print(model.evaluate(x_test, y_test, batch_size = batch_size))
    return history, model, tokenizer

def prediction(DATA_DIR, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS):
    '''
    Function used to take the lstm model and use it on the test set for predictions
    '''
    histo, model, tokenizer = lstm('train.csv', MAX_SEQUENCE_LENGTH = 1000, MAX_NB_WORDS = 10000, 
                 output_dim = 256, units = 128, kernel_regularizer = l2(0), 
                 activity_regularizer=l2(0), dropout = 0.5, classes = 3, 
                 activation_function = 'softmax', loss = 'categorical_crossentropy',
                 learning_rate=0.001, metric=['accuracy'], batch_size=100, epochs=2)

    plt.plot(histo.history['acc'])
    plt.plot(histo.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(histo.history['loss'])
    plt.plot(histo.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()  
    
    test = pd.read_csv(DATA_DIR)
    len(test)
    texts = test['text']
    
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict_proba(np.array(data))

    result = pd.DataFrame()
    test_id = test['id']
    result['id'] = test_id
       
    result['EAP'] = [x[0] for x in prediction]
    result['HPL'] = [x[1] for x in prediction]
    result['MWS'] = [x[2] for x in prediction]
     
    return result.to_csv("result.csv", index=False)
def main():    
    prediction('test.csv', MAX_SEQUENCE_LENGTH = 1000, MAX_NB_WORDS = 10000)
if __name__ == "__main__":
    main()    
