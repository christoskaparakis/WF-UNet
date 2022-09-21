import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation, multiply, Lambda, Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Permute, Concatenate, Conv2D, Add
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, DepthwiseConv2D, LayerNormalization, Softmax
from tensorflow.keras import backend as K
from utils import *
    
    
# def UNet_original(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

#     #--- Contracting part / encoder ---#
#     inputs = Input(shape = (lags, latitude, longitude, features)) 
#     conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
#     conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
#     pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
#     conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
#     conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
#     pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
#     conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
#     conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
#     pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
#     conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
#     conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
#     drop4 = Dropout(dropout)(conv4)
    
#     #--- Bottleneck part ---#
#     pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
#     conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
#     conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv5)
#     compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
#     drop5 = Dropout(dropout)(compressLags)
    
#     #--- Expanding part / decoder ---#
#     up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
#     compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
#     merge6 = concatenate([compressLags,up6], axis = -1)
#     conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
#     conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

#     up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
#     compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
#     merge7 = concatenate([compressLags,up7], axis = -1)
#     conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
#     conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

#     up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
#     compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
#     merge8 = concatenate([compressLags,up8], axis = -1)
#     conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
#     conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

#     up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
#     compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
#     merge9 = concatenate([compressLags,up9], axis = -1)
#     conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
#     conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
#     conv10 = Conv3D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

#     return Model(inputs = inputs, outputs = conv10)

def UNet_original(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (latitude, longitude, lags)) 
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = Conv2D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv2D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = -1)
    conv6 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv2D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = -1)
    conv7 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv2D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = -1)
    conv8 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv2D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = -1)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv9 = Conv2D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv2D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)
    
    
    
# The data has 3-dimensional shape in the encoder and 2-dimensional shape in the decoder
def UNet_3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)
    
    

# Includes residual connections and more consecutive convolutional operations
def UNet_Res3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
    
    def residual_block(x, f, k):
        shortcut=x
        #First component
        x = Conv3D(f, k, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #Second component
        x = Conv3D(f, k, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = BatchNormalization()(x)
        #Addition
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x     
        
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = residual_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = residual_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = residual_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = residual_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = residual_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = residual_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = residual_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = residual_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = residual_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)



# Includes parallel convolutions and residual connections
def UNet_InceptionRes3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
       
    def res_inception_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x = concatenate([x1, x2, x3], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = res_inception_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = res_inception_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = res_inception_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = res_inception_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = res_inception_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = res_inception_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = res_inception_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = res_inception_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = res_inception_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)



# Includes parallel convolutions, asymmetric convolutions and residual connections
def UNet_AsymmetricInceptionRes3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

    def res_inception_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x2 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x4 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x4 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x5 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x5 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x5 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = res_inception_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = res_inception_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = res_inception_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = res_inception_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = res_inception_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = res_inception_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = res_inception_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = res_inception_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = res_inception_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)

def SmaAt_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):      

    def channel_attention(input_feature, ratio=16):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling2D()(input_feature)    
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        
        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        
        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature):
        kernel_size = 7
        
        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2,3,1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature
        
        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters = 1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)	
        assert cbam_feature.shape[-1] == 1
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
            
        return multiply([input_feature, cbam_feature])

    def cbam_block(cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """
        
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature


    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = DepthwiseConv2D(4*filters, 2,padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = DepthwiseConv2D(4*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    cbam1 = cbam_block(conv1)
    
    conv2 = DepthwiseConv2D(8*filters, 3,padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = DepthwiseConv2D(8*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    cbam2 = cbam_block(conv2)
    
    conv3 = DepthwiseConv2D(16*filters, 3,padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = DepthwiseConv2D(16*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    cbam3 = cbam_block(conv3)

    conv4 = DepthwiseConv2D(32*filters, 3,padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = DepthwiseConv2D(32*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(conv4)
    cbam4 = cbam_block(conv4)
    
    #--- Bottleneck part ---#
    conv5 = DepthwiseConv2D(32*filters, 3,padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = DepthwiseConv2D(32*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    cbam5 = cbam_block(conv5)
    # compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    # drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = DepthwiseConv2D(32*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(cbam5))
    # compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([cbam4,up6], axis = -1)
    conv6 = DepthwiseConv2D(16*filters, 3,padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = DepthwiseConv2D(16*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = DepthwiseConv2D(16*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    merge7 = concatenate([cbam3,up7], axis = -1)
    conv7 = DepthwiseConv2D(8*filters, 3,padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = DepthwiseConv2D(8*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = DepthwiseConv2D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    merge8 = concatenate([cbam2,up8], axis = -1)
    conv8 = DepthwiseConv2D(4*filters, 3,padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = DepthwiseConv2D(4*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = DepthwiseConv2D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    merge9 = concatenate([cbam1,up9], axis = -1)
    conv9 = DepthwiseConv2D(4*filters, 3,padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = DepthwiseConv2D(4*filters, 3,padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)

def broad_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

    def image_level_feature_pooling(x, f):
        up_size = x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size = (1,up_size[0],up_size[1]))(x)
        return x


    def ASPP_block(x, f):
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=6,  kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=12,  kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=18,  kernel_initializer = kernel_init)(x)
        x5 = image_level_feature_pooling(x, f)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        return x


    def convolutional_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x2 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x4 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x4 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x5 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x5 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x5 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = convolutional_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = convolutional_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = convolutional_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = convolutional_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)

    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    aspp = ASPP_block(compressLags, 16*filters)
    drop5 = Dropout(dropout)(aspp)

    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    # print('drop5 ' + str(drop5.shape))
    # print('up6 ' + str(up6.shape))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv4)
    # print('drop4 ' + str(drop4.shape))
    # print('compressLags ' + str(compressLags.shape))
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)


    # Adding a convolutional layer with 8 times the number of features as the output.
    conv10 = Conv3D(features_output, 1, activation = 'relu', padding = 'same')(conv9) #Reduce last dimension     

    return Model(inputs = inputs, outputs = conv10)


def new_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def cbam_block(cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """
        
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        cbam_feature = Activation('relu')(cbam_feature)
        return cbam_feature

    def channel_attention(input_feature, ratio=8):
        
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling3D()(input_feature)    
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        
        max_pool = GlobalMaxPooling3D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        
        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature):
        kernel_size = 7
        
        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2,3,1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature
        
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters = 1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)	
        assert cbam_feature.shape[-1] == 1
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
            
        return multiply([input_feature, cbam_feature])

    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = BatchNormalization()(conv1)
    cbam1 = cbam_block(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(cbam1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = BatchNormalization()(conv2)
    cbam2 = cbam_block(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(cbam2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = BatchNormalization()(conv3)
    cbam3 = cbam_block(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(cbam3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = BatchNormalization()(conv4)
    cbam4 = cbam_block(conv4)
    drop4 = Dropout(dropout)(cbam4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = BatchNormalization()(conv5)
    cbam5 = cbam_block(conv5)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(cbam5)
    compressLags = BatchNormalization()(compressLags)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    compressLags = BatchNormalization()(compressLags)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = BatchNormalization()(conv6)
    cbam6 = cbam_block(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(cbam6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    compressLags = BatchNormalization()(compressLags)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = BatchNormalization()(conv7)
    cbam7 = cbam_block(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(cbam7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    compressLags = BatchNormalization()(compressLags)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = BatchNormalization()(conv8)
    cbam8 = cbam_block(conv8)
    
    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(cbam8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    compressLags = BatchNormalization()(compressLags)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = BatchNormalization()(conv9)
    cbam9 = cbam_block(conv9)

    conv9 = Conv3D(2*features_output, 3, padding = 'same', kernel_initializer = kernel_init)(cbam9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv3D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)

def exp_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def squeeze_excite_block(input_tensor, ratio=16):
        """ Create a channel-wise squeeze-excite block
        Args:
            input_tensor: input Keras tensor
            ratio: number of output filters
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        """
        init = input_tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = tensor_shape(init)[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x

    def channel_attention(input_feature, ratio=8):
        
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling3D()(input_feature)    
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        
        max_pool = GlobalMaxPooling3D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        
        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
        return multiply([input_feature, cbam_feature])

    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = residual_block(conv1, filters, 3)
    ca1 = channel_attention(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = residual_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = residual_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = residual_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = residual_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = residual_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = residual_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = residual_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = residual_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension 


def MLF_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def UNet(x, lags,features_output, filters, dropout, kernel_init):
        
        #--- Contracting part / encoder ---#
        conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
        pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
        
        conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
        conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
        pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
        
        conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
        conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
        pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
        
        conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
        conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
        drop4 = Dropout(dropout)(conv4)
        
        #--- Bottleneck part ---#
        pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
        conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
        compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
        conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
        drop5 = Dropout(dropout)(conv5)
        
        #--- Expanding part / decoder ---#
        up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
        compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
        merge6 = concatenate([compressLags,up6], axis = -1)
        conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
        conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

        up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
        compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
        merge7 = concatenate([compressLags,up7], axis = -1)
        conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
        conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

        up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
        compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
        merge8 = concatenate([compressLags,up8], axis = -1)
        conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
        conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

        up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
        compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
        merge9 = concatenate([compressLags,up9], axis = -1)
        conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
        conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
        conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
        
        conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension

        return Model(inputs = x, outputs = conv10)

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))

    streamA = UNet(inputA, lags, features_output, filters, dropout, kernel_init)
    streamB = UNet(inputB, lags, features_output, filters, dropout, kernel_init)

    fusion = concatenate([streamA.output, streamB.output])

    out = Conv3D(features_output, 1, activation = 'linear', padding = 'same')(fusion) #Reduce last dimension    
    return Model(inputs = [inputA,inputB], outputs = out)

def MLF_UNet_triple(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def UNet(x, lags,features_output, filters, dropout, kernel_init):
        
        #--- Contracting part / encoder ---#
        conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
        pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
        
        conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
        conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
        pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
        
        conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
        conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
        pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
        
        conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
        conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
        drop4 = Dropout(dropout)(conv4)
        
        #--- Bottleneck part ---#
        pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
        conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
        compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
        conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
        drop5 = Dropout(dropout)(conv5)
        
        #--- Expanding part / decoder ---#
        up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
        compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
        merge6 = concatenate([compressLags,up6], axis = -1)
        conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
        conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

        up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
        compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
        merge7 = concatenate([compressLags,up7], axis = -1)
        conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
        conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

        up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
        compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
        merge8 = concatenate([compressLags,up8], axis = -1)
        conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
        conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

        up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
        compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
        merge9 = concatenate([compressLags,up9], axis = -1)
        conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
        conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
        conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
        
        conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension

        return Model(inputs = x, outputs = conv10)

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))
    inputC = Input(shape = (lags, latitude, longitude, features))

    streamA = UNet(inputA, lags, features_output, filters, dropout, kernel_init)
    streamB = UNet(inputB, lags, features_output, filters, dropout, kernel_init)
    streamC = UNet(inputC, lags, features_output, filters, dropout, kernel_init)

    fusion = concatenate([streamA.output, streamB.output, streamC.output])

    out = Conv3D(features_output, 1, activation = 'linear', padding = 'same')(fusion) #Reduce last dimension    
    return Model(inputs = [inputA,inputB], outputs = out)

def MEF_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))

    fusion = concatenate([inputA, inputB])

    #--- Contracting part / encoder ---#
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(fusion)
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9)

    return Model(inputs = [inputA,inputB], outputs = conv10)

def MEF_UNet_triple(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))
    inputC = Input(shape = (lags, latitude, longitude, features))

    fusion = concatenate([inputA, inputB, inputC])

    #--- Contracting part / encoder ---#
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(fusion)
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9)

    return Model(inputs = [inputA,inputB], outputs = conv10)


def MIF_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))

    ##################################################
    #--- Contracting part / encoder of variable 1 ---#
    conv11 = Conv3D(filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputA)
    conv11 = Conv3D(filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv11)
    pool11 = MaxPool3D(pool_size=(1, 2, 2))(conv11)
    
    conv21 = Conv3D(2*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool11)
    conv21 = Conv3D(2*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv21)
    pool21 = MaxPool3D(pool_size=(1, 2, 2))(conv21)
    
    conv31 = Conv3D(4*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool21)
    conv31 = Conv3D(4*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv31)
    pool31 = MaxPool3D(pool_size=(1, 2, 2))(conv31)
    
    conv41 = Conv3D(8*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool31)
    conv41 = Conv3D(8*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv41)
    drop41 = Dropout(dropout)(conv41)
    
    pool41 = MaxPool3D(pool_size=(1, 2, 2))(drop41)
    conv51 = Conv3D(16*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool41)
    compressLags1 = Conv3D(16*filters//2, (lags,1,1),activation = 'relu', padding = 'valid')(conv51)
    conv51 = Conv3D(16*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags1)
    
    ##################################################

    ##################################################
    #--- Contracting part / encoder of variable 2 ---#
    conv12 = Conv3D(filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputB)
    conv12 = Conv3D(filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv12)
    pool12 = MaxPool3D(pool_size=(1, 2, 2))(conv12)
    
    conv22 = Conv3D(2*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool12)
    conv22 = Conv3D(2*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv22)
    pool22 = MaxPool3D(pool_size=(1, 2, 2))(conv22)
    
    conv32 = Conv3D(4*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool22)
    conv32 = Conv3D(4*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv32)
    pool32 = MaxPool3D(pool_size=(1, 2, 2))(conv32)
    
    conv42 = Conv3D(8*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool32)
    conv42 = Conv3D(8*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv42)
    drop42 = Dropout(dropout)(conv42)
    
    pool42 = MaxPool3D(pool_size=(1, 2, 2))(drop42)
    conv52 = Conv3D(16*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool42)
    compressLags2 = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv52)
    conv52 = Conv3D(16*filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags2)
    
    ##################################################

    ##################################################
    #--- Expanding part / decoder ---#
    #--- Expanding part / decoder ---#
    fusion1 = concatenate([conv11,conv12], axis = -1)
    # fusion1 = Conv3D(fusion1.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion1)

    fusion2 = concatenate([conv21,conv22], axis = -1)
    # fusion2 = Conv3D(fusion2.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion2)

    fusion3 = concatenate([conv31,conv32], axis = -1)
    # fusion3 = Conv3D(fusion3.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion3)

    fusion4 = concatenate([conv41,conv42], axis = -1)
    # fusion4 = Conv3D(fusion4.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion4)
    fusion4 = Dropout(dropout)(fusion4)

    fusion5 = concatenate([conv51,conv52], axis = -1)
    # fusion5 = Conv3D(fusion5.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion5)
    fusion5 = Dropout(dropout)(fusion5)

    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(fusion5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9)

    return Model(inputs = [inputA,inputB], outputs = conv10)

def MIF_UNet_triple(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    inputA = Input(shape = (lags, latitude, longitude, features))
    inputB = Input(shape = (lags, latitude, longitude, features))
    inputC = Input(shape = (lags, latitude, longitude, features))

    ##################################################
    #--- Contracting part / encoder of variable 1 ---#
    conv11 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputA)
    conv11 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv11)
    pool11 = MaxPool3D(pool_size=(1, 2, 2))(conv11)
    
    conv21 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool11)
    conv21 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv21)
    pool21 = MaxPool3D(pool_size=(1, 2, 2))(conv21)
    
    conv31 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool21)
    conv31 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv31)
    pool31 = MaxPool3D(pool_size=(1, 2, 2))(conv31)
    
    conv41 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool31)
    conv41 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv41)
    drop41 = Dropout(dropout)(conv41)
    
    pool41 = MaxPool3D(pool_size=(1, 2, 2))(drop41)
    conv51 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool41)
    compressLags1 = Conv3D(16*filters//3, (lags,1,1),activation = 'relu', padding = 'valid')(conv51)
    conv51 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags1)
    
    ##################################################

    ##################################################
    #--- Contracting part / encoder of variable 2 ---#
    conv12 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputB)
    conv12 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv12)
    pool12 = MaxPool3D(pool_size=(1, 2, 2))(conv12)
    
    conv22 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool12)
    conv22 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv22)
    pool22 = MaxPool3D(pool_size=(1, 2, 2))(conv22)
    
    conv32 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool22)
    conv32 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv32)
    pool32 = MaxPool3D(pool_size=(1, 2, 2))(conv32)
    
    conv42 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool32)
    conv42 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv42)
    drop42 = Dropout(dropout)(conv42)
    
    pool42 = MaxPool3D(pool_size=(1, 2, 2))(drop42)
    conv52 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool42)
    compressLags2 = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv52)
    conv52 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags2)
    
    ##################################################

    ##################################################
    #--- Contracting part / encoder of variable 3 ---#
    conv13 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputC)
    conv13 = Conv3D(filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv13)
    pool13 = MaxPool3D(pool_size=(1, 2, 2))(conv13)
    
    conv23 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool13)
    conv23 = Conv3D(2*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv23)
    pool23 = MaxPool3D(pool_size=(1, 2, 2))(conv23)
    
    conv33 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool23)
    conv33 = Conv3D(4*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv33)
    pool33 = MaxPool3D(pool_size=(1, 2, 2))(conv33)
    
    conv43 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool33)
    conv43 = Conv3D(8*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv43)
    drop43 = Dropout(dropout)(conv43)
    
    pool43 = MaxPool3D(pool_size=(1, 2, 2))(drop43)
    conv53 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool43)
    compressLags3 = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv53)
    conv53 = Conv3D(16*filters//3, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags3)
    
    ##################################################

    ##################################################
    #--- Expanding part / decoder ---#
    #--- Expanding part / decoder ---#
    fusion1 = concatenate([conv11,conv12,conv13], axis = -1)
    # fusion1 = Conv3D(fusion1.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion1)

    fusion2 = concatenate([conv21,conv22,conv23], axis = -1)
    # fusion2 = Conv3D(fusion2.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion2)

    fusion3 = concatenate([conv31,conv32,conv33], axis = -1)
    # fusion3 = Conv3D(fusion3.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion3)

    fusion4 = concatenate([conv41,conv42,conv43], axis = -1)
    # fusion4 = Conv3D(fusion4.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion4)
    fusion4 = Dropout(dropout)(fusion4)

    fusion5 = concatenate([conv51,conv52,conv53], axis = -1)
    # fusion5 = Conv3D(fusion5.shape[-1]//2, 2, activation = 'relu', padding = 'same')(fusion5)
    fusion5 = Dropout(dropout)(fusion5)

    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(fusion5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(fusion1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9)

    return Model(inputs = [inputA,inputB], outputs = conv10)