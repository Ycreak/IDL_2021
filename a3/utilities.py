import pickle
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics as crf_metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import argparse

from datetime import datetime

def pickle_write(path, file_name, object):
    destination = path + file_name
    with open(destination, 'wb') as f:
        pickle.dump(object, f)

def pickle_read(path, file_name):
    destination = path + file_name
    with open(destination, 'rb') as f:
        return pickle.load(f)

def create_confusion_matrix(model, X, y):
    """Creates a confusion matrix from the given model, X and y sets. As y is one-hot encoded, we need to take
    the argmax value. Secondly, because the list of lists structure (sentences), we need to flatten both prediction
    lists in order to pass them to the confusion_matrix function.

    Args:
        model (object): of lstm model
        X (list): list of lists with sentences encoded as integers
        y (list): of labels, same as X.

    Returns:
        confusion matrix: of labels
    """        
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.argmax(y, axis=-1)

    flat_list = [item for sublist in y for item in sublist]
    flat_list2 = [item for sublist in y_pred for item in sublist]

    return confusion_matrix(flat_list, flat_list2)

def create_metrics_report(model, X, y):
    """Returns a metrics classification report given the model and X and y sets.
    This shows the precision and recall of label predictions

    Args:
        model (object): the given model, LSTM in this case
        X (list): with training examples
        y (list): with training labels

    Returns:
        dataframe: with the metrics report to be printed
    """        
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.argmax(y, axis=-1)

    metrics_report = crf_metrics.flat_classification_report(
        y, y_pred, labels=[0,1,2], target_names=['long', 'short', 'elision'], digits=4
    )

    return metrics_report        

def create_line_plot(plots, ylabel, xlabel, plot_titles, title, plotname):
    # Simple function to easily create plots
    path = './plots/'
    time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    full_file_name = '{0}{1}_{2}.png'.format(path, plotname, time)
    
    for plot_line in plots:
        plt.plot(plot_line)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(plot_titles, loc='lower right')
    plt.title(title)
    plt.savefig(full_file_name)
    # plt.show()
    plt.clf()    

def create_heatmap(dataframe,xlabel, ylabel, title, filename, vmax):
    # Simple function to create a heatmap

    # dataframe.to_numpy().max()

    path = './plots/'
    time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    full_file_name = '{0}{1}_{2}.png'.format(path, filename, time)

    sn.set(font_scale=1, rc = {'figure.figsize':(12,8)})
    sn.heatmap(dataframe, annot=True, fmt='g', annot_kws={"size": 10}, cmap='Blues', vmin=0, vmax=vmax)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(full_file_name, bbox_inches='tight')        
    plt.clf()  

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x    