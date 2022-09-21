import tensorflow as tf
import numpy as np
from os import listdir

class DataGeneratorPrecipitationData(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, lags, steps_ahead=1, multimodal=False, multistream=False, triplestream=False):
        self.data = data
        self.b_size = batch_size
        self.lags = lags
        self.multimodal = multimodal
        self.multistream = multistream
        self.triplestream = triplestream
        self.time_steps = data[0].shape[0]
        self.steps_ahead = steps_ahead-1

    
    #Calculates the number of batches: samples/batch_size
    def __len__(self):
        #Calculating the number of batches 
        return int(self.time_steps/self.b_size)

    
    #Obtains one batch of data 
    def __getitem__(self, idx):
        if self.multimodal:
            x = self.data[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :,:]
            y = self.data[idx*self.b_size:(idx+1)*self.b_size, self.lags+self.steps_ahead, :, :, 0]
            
            # x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
            y = np.expand_dims(y, axis=1)
        elif self.multistream:
            data1 = self.data[0]
            x1 = data1[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            x1 = np.expand_dims(x1, axis=-1)

            data2 = self.data[1]
            x2 = data2[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            x2 = np.expand_dims(x2, axis=-1)

            x = [x1,x2]
            
            y = data1[idx*self.b_size:(idx+1)*self.b_size, self.lags+self.steps_ahead, :, :]
            y = np.expand_dims(y, axis=-1)
            y = np.expand_dims(y, axis=1)
        elif self.triplestream:
            data1 = self.data[0]
            x1 = data1[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            x1 = np.expand_dims(x1, axis=-1)

            data2 = self.data[1]
            x2 = data2[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            x2 = np.expand_dims(x2, axis=-1)

            data3 = self.data[2]
            x3 = data2[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            x3 = np.expand_dims(x2, axis=-1)

            x = [x1,x2,x3]
            
            y = data1[idx*self.b_size:(idx+1)*self.b_size, self.lags+self.steps_ahead, :, :]
            y = np.expand_dims(y, axis=-1)
            y = np.expand_dims(y, axis=1)
        else:
            x = self.data[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
            y = self.data[idx*self.b_size:(idx+1)*self.b_size, self.lags+self.steps_ahead, :, :]
            
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
            y = np.expand_dims(y, axis=1)
                
        return x, y