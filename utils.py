import tensorflow as tf
from kapre.time_frequency import STFT, Magnitude, MagnitudeToDecibel
from kapre.signal import LogmelToMFCC
from kapre.augmentation import SpecAugment
from tensorflow.keras.layers import Input, Concatenate
import tensorflow_hub as hub
import scipy as sp
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_processor():
    input_ = Input(shape=(64000, 1))
    fourier = STFT(n_fft=2048,
                 win_length=128,
                 hop_length=64,
                 pad_end=False,
                 input_data_format='channels_last',
                 output_data_format='channels_last',
                 name='stft-layer')(input_)

    magnitude = Magnitude()(fourier)
    mag_to_db = MagnitudeToDecibel()(magnitude)

    mfcc_layer = LogmelToMFCC(n_mfccs=1025)(mag_to_db)

    augmenation_layer = SpecAugment(
                freq_mask_param=10,
                time_mask_param=20,
                n_freq_masks=2,
                n_time_masks=3,
                mask_value=-70)(mag_to_db)

    concat_output = Concatenate(axis=-1)([augmenation_layer, mfcc_layer])

    processor = tf.keras.Model(inputs=[input_], outputs=[concat_output], name='processor_network')

    return processor



def load_wav_16k_mono(filepath, expand_dim=True):
    sample_rate, wav_data = wavfile.read(filepath, mmap=False) # sp.io.wavfile.read()
    
    # convert to mono
    if wav_data.ndim == 2:
        wav_data = np.sum(wav_data, axis=1)
    
    # resample
    if sample_rate != 16000:
        desired_length = int(round(float(len(wav_data)) /
                                   sample_rate * 16000))
        wav_data = sp.signal.resample(wav_data, desired_length)
        
    # convert to tensor
    wav = tf.convert_to_tensor(wav_data, tf.float32)
    
    # trim, normalize, pad, format
    wav = wav[:64000]
    wav = tf.math.divide(
        wav,
        tf.math.reduce_max(tf.math.abs(wav)))
    zero_padding = tf.zeros(shape=([64000] - tf.shape(wav)), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], axis=0)
    if expand_dim == True:
        wav = tf.expand_dims(wav, axis=1)

    return wav
    

def build_yamnet_base():
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    return yamnet_model


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 training,
                 output_format, 
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.output_format = output_format # 'wave' or 'spec_mfcc'
        
        self.n = self.df.shape[0]
        self.n_classes = df[y_col].nunique()
        
        self.processor = build_processor()
        self.loader = load_wav_16k_mono
        #self.netprocessor = self.build_processor()
    
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

       
    def __get_input(self, filepath):
        return self.loader(filepath)
        
        
    # def __get_output(self, label):
    #     return tf.cast(label, tf.float32)
    
    
    def __get_data(self, batches):
        path_batch = batches[self.X_col]
        wav_batch = tf.convert_to_tensor([self.__get_input(path) for path in path_batch], dtype=tf.float32)
        
        y_batch = batches[self.y_col].values
        
        # Here we call the processor and convert batches of wavs to batches of 2-d channel spectrogram/mfcc images
        if self.output_format == 'spec_mfcc':
            X_batch = self.processor(wav_batch, training=self.training)
            return X_batch, y_batch  
    
        elif self.output_format == 'wave':
            return tf.squeeze(wav_batch, axis=2), y_batch
    
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)    
        return X, y
    

    def __len__(self):
        return self.n // self.batch_size



class_ids = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

#n_classes = np.unique(long_clips.classID).shape[0]





# be sure to check the parameters of the methods
# think about the order that they will be called in!

class SoundClip:
    '''
    Docstring: Class for single datapoint input by the user, let's do stuff with it
    It all starts with a path to an uploaded file on the server
    '''
    def __init__(self) -> None:
        pass

    def run_processor(self, waveform):
        pass

    def predict_single(self, filepath):
        pass

    def plot_spectrogram(self, waveform):
        pass

    def plot_mfcc(self, waveform):
        pass



if __name__ == '__main__':
    print('utils as main - you should not see this')