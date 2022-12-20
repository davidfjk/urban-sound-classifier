import requests
import datetime
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Dropout, MaxPooling2D, Concatenate, Lambda, BatchNormalization
from tensorflow.keras.layers import Conv1D, LSTM, GRU, TimeDistributed, LayerNormalization, GlobalMaxPooling2D
import tensorflow.keras.backend as K


# define data download function, define model architectures, import datagenerator
# compile, train and persist the models (with timestamp)
# we will not run this locally ourselves, just for documentation and reproducability


def download_urbansound8k():
    '''
    Beware: this will start a download of 6-7 GB of sound clip files, unzip the contents
    '''
    pass

def get_metadata_clean():
    # all the cleaning of the metadata.csv happens here
    pass

def get_data_partitions():
    # train-valid-test split, discard clips which are too short, check kaggle notebook for sepcifics (e.g. seed)
    pass






if __name__ == '__main__':

    # assert that GPU is activated
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found - model training not advised')
    else:
        print('Found GPU at: {}'.format(device_name))

    # start the compilation and training cycle of the model that was passed as an argument with argparse
    # also give user some flexibility of how to split data partitions etc. with argparse