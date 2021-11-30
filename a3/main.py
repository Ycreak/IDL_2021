import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose

import pandas as pd
# Class imports
import data_generation as dat_gen
import utilities as util


def load_data(create_dataset):
    if create_dataset:
        X_text, X_img, y_text, y_img = dat_gen.create_data()         
        util.pickle_write('./pickle/', 'X_text.pickle', X_text)
        util.pickle_write('./pickle/', 'X_img.pickle', X_img)
        util.pickle_write('./pickle/', 'y_text.pickle', y_text)
        util.pickle_write('./pickle/', 'y_img.pickle', y_img)
    else:
        X_text = util.pickle_read('./pickle/', 'X_text.pickle')
        X_img = util.pickle_read('./pickle/', 'X_img.pickle')
        y_text = util.pickle_read('./pickle/', 'y_text.pickle')
        y_img = util.pickle_read('./pickle/', 'y_img.pickle')

    return X_text, X_img, y_text, y_img 

## Display the samples that were created
def display_sample(n):
    labs = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        plt.axis('off')
        plt.title(labs[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nSample ID: {n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

def encode_labels(labels, max_len=4):
    n = len(labels)
    length = len(labels[0])
    char_map = dict(zip(unique_characters, range(len(unique_characters))))
    one_hot = np.zeros([n, length, len(unique_characters)])
    for i, label in enumerate(labels):
        m = np.zeros([length, len(unique_characters)])
        for j, char in enumerate(label):
            m[j, char_map[char]] = 1
        one_hot[i] = m

    return one_hot 

def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])
    return predicted

def get_model():
    # We start by initializing a sequential model
    model = tf.keras.Sequential()
    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [7, 13] as we have queries of length 7 and 13 unique characters. 
    # Each of these 7 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 7 elements). # Hint: In other applications, where your input sequences 
    # have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    model.add(LSTM(256, input_shape=(max_query_length, len(unique_characters))))
    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 4 times as that's the maximum length of the output (e.g. '  1-199' = '-198')
    # when using 3-digit integers in queries. In other words, the RNN will always produce 4 characters as its output.
    model.add(RepeatVector(max_answer_length))
    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). 
    # This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    model.add(LSTM(128, return_sequences=True))
    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    model.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))
    # Next we compile the model using categorical crossentropy as our loss function.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def fit_model(model, X, y, epochs):
    history = model.fit(X, y, batch_size=32, epochs=epochs, verbose=1)
    # Save the model for loading at a later time
    model.save('./model')
    return model, history

if __name__ == "__main__":

    # Argument parser.
    p = argparse.ArgumentParser()
    p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--create_dataset", action="store_true", help="specify whether to create the dataset: if not specified, we load from disk")
    p.add_argument("--text2text", action="store_true", help="specify whether to run the text2text model")
    p.add_argument("--img2text", action="store_true", help="specify whether to run the img2text model")

    FLAGS = p.parse_args()

    unique_characters = '0123456789+- ' # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
    highest_integer = 199 # Highest value of integers contained in the queries
    max_int_length = len(str(highest_integer)) # 
    max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
    max_answer_length = max_int_length + 1    # Maximum length of the answer string
    
    # Booleans
    num_epochs = 25
    create_line_plot = True
    evaluate = False

    # Create the data (might take around a minute)
    X_text, X_img, y_text, y_img = load_data(create_dataset=FLAGS.create_dataset)
    # print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)      

    X_text_onehot = encode_labels(X_text) # columns are labels, rows are X_text chars
    y_text_onehot = encode_labels(y_text)

    # print(X_text_onehot.shape, y_text_onehot.shape)
    # print(X_text[5578], X_text_onehot[5578], y_text[5578])  
    # display_sample(5578)  

    #############################
    # 1. Text-to-text RNN model #
    #############################
    if FLAGS.text2text:
        X_train, X_test, y_train, y_test = train_test_split(X_text_onehot, y_text_onehot, test_size=0.2)
        
        if FLAGS.create_model:
            model = get_model()
            model, history = fit_model(model, X_train, y_train, num_epochs)

            if create_line_plot:
                util.create_line_plot( # Create a line plot of the training
                    plots = (history.history['accuracy'],),
                    ylabel = 'Accuracy',
                    xlabel = 'Epoch',
                    plot_titles = ['train'],
                    title = 'LSTM accuracy',
                    plotname = 'lstm_accuracy'
                )
        else:
            model = tf.keras.models.load_model('./model')

        if evaluate:
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            print('Model accuracy: {0}. Model loss: {1}.'.format(accuracy, loss))

        # Create a confusion matrix for the labels predictions
        confusion_matrix = util.create_confusion_matrix(model, X_test, y_test)
        label_list = ['0','1','2','3','4','5','6','7','8','9','-','space'] #TODO: y does not predict +. can i just remove
        df_confusion_matrix = pd.DataFrame(confusion_matrix, index = label_list,
                                        columns = label_list)

        util.create_heatmap(dataframe = df_confusion_matrix,
                            ylabel =  'PREDICTED',
                            xlabel = 'TRUTH', 
                            title = 'CONFUSION MATRIX TEXT2TEXT',
                            filename = 'confusion_matrix_lstm_text2text',
                            vmax = 500
                            )
    ##############################
    # 2. Image-to-text RNN model #
    ##############################    
    if FLAGS.img2text:
        
        # We have 7 MNIST digits for every X_img example. Each MNIST image is 28x28 pixels. Our input dimensions are therefore 28 * 7(28) = 28 * 196 = 5488
        x_dim = 28 * 7
        y_dim = 28 * 1
        n_inputs = x_dim * y_dim # maximum query length

        n_features = X_img.shape[1] # Count the number of columsn. these are our features
        X_img = X_img.reshape(80000, 5488, 1).astype('float32')

        print('SHAPE', X_img.shape)

        print(n_features)
        print(len(X_img))
        # exit(0)

        max_query_length = n_inputs # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
        max_answer_length = max_int_length + 1    # Maximum length of the answer string
        
        num_timesteps = x_dim * y_dim
        num_features = len(unique_characters)
        # print(y_text_onehot.shape)
        # y_text_onehot = y_text_onehot.reshape(1,-1)
        # print(y_text_onehot.shape) # (3,)

        # exit(0)

        # x = tf.placeholder("float", [None, 196, 28])
        # y = tf.placeholder("float", [None, 12])
        # print(X_img[5567])
        # print(type(X_img[5567]))
        # # display_sample(5567)

        # print(X_img[5567].shape)

        # We start by initializing a sequential model
        model = tf.keras.Sequential()
        # "Encode" the input sequence using an RNN, producing an output of size 256.
        # In this case the size of our input vectors is [7, 13] as we have queries of length 7 and 13 unique characters. 
        # Each of these 7 elements in the query will be fed to the network one by one,
        # as shown in the image above (except with 7 elements). # Hint: In other applications, where your input sequences 
        # have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
        model.add(LSTM(256, input_shape=(num_timesteps, num_features)))
        # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 4 times as that's the maximum length of the output (e.g. '  1-199' = '-198')
        # when using 3-digit integers in queries. In other words, the RNN will always produce 4 characters as its output.
        model.add(RepeatVector(max_answer_length))
        # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). 
        # This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
        model.add(LSTM(128, return_sequences=True))
        # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
        model.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))
        # Next we compile the model using categorical crossentropy as our loss function.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        history = model.fit(X_img, y_text_onehot, batch_size=32, epochs=2, verbose=1)











    # print('#####')
    # print(X_text[5578])
    # print('#####')
    # print(X_text_onehot[5578])
    # print('#####')
    # print(y_text_onehot[5567])
    # print('#####')
    # print(y_text[5578])  
    # print('#####')