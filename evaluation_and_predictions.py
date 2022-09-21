import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from models import *
from generators import DataGeneratorPrecipitationData as DataGenerator


##---- Evaluating model saved in "saved_models_precipitation/best_model.hdf5" ----##

# Select dataset
# filename = "dataset_precipitation/Data_20/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_20.h5"
filename = (
    "final_datasets/train_test_2016-2021_input-length_12_img-ahead_6_rain-threshhold_50.h5"
)


# Read dataset
try:
	f = h5py.File(filename, 'r')
except:
	raise Exception('\n\nNo data was found! Get and decompress the data as indicated first.')
    
max_train_val = 0.024250908
# Test data to numpy array
tp_test = f["/test/tp_images"][1790:1810,:,:96,:96]  # remove the slicing later
data_test = tp_test/max_train_val #normalize data

data_wind_max = 42.895767
data_test1 = f["/test/u100_images"][1790:1810,:,:96,:96]
data_test2 = f["/test/v100_images"][1790:1810,:,:96,:96]
data_wind_speed = np.sqrt((data_test1**2)+data_test2**2)
del data_test1
del data_test2

data_wind_speed_test = data_wind_speed/data_wind_max


#Parameters
lags = 12
lat = data_test.shape[-2]
long = data_test.shape[-1]
loss = 'mse'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
batch_size = 2


# Mean value training dataset 50% pixel occurence
threshold = 0.0047769877


# Custom metrics (Denormalized MSE and binarized metrics)
denormalized_mse = MSE_denormalized(max_train_val, batch_size, reduction_sum=True, latitude=lat, longitude=long) #MSE per image
binarized_metrics = thresholded_mask_metrics(threshold=threshold)
metrics = [denormalized_mse.mse_denormalized_per_image, denormalized_mse.mse_denormalized_per_pixel, binarized_metrics.acc, binarized_metrics.precision, binarized_metrics.recall, binarized_metrics.f1_score, binarized_metrics.CSI, binarized_metrics.FAR]


#Loade model and compile with custom metrics
filepath="saved_models_precipitation/best_model_rain_wind_mlf_1_ahead.hdf5"
try:
    # model = broad_UNet(12, 96, 96, 1, 1, 2, 0.5)
    # model = UNet_original(12, 96, 96, 1, 1, 16, 0.5)
    model = MLF_UNet(12, 96, 96, 1, 1, 16, 0.5)
    model.load_weights(filepath)
	# model = load_model(filepath, compile=False)
except:
	raise Exception('\n\nNo trained model was found! Run first the trainig script or request pretrained model.')
model.compile(loss=denormalized_mse.mse_denormalized_per_image, optimizer=optimizer, metrics=metrics)


#Generator
test_generator = DataGenerator([data_test,data_wind_speed_test], batch_size, lags,multistream=True, steps_ahead=4)


# #Evaluate with generator 
# print("\nEvaluating...")
# result = model.evaluate(test_generator)

# print("\n>>> Results evaluation:")
# print(" - MSE:", result[2])
# print(" - MSE per image:", result[1])
# print(" - Acc:", result[3])
# print(" - Precision:", result[4])
# print(" - Recall:", result[5])



for t in range(len(data_test)):
    print(t)
    #Generating targets and labels
    x = data_test[t:(t+2), :lags, :, :]
    x = np.expand_dims(x, axis=-1)

    

    x1 = data_wind_speed_test[t:(t+2), :lags, :, :]
    x1 = np.expand_dims(x1, axis=-1)
    x=[x,x1]

    y = data_test[t:(t+2), lags+1-1, :, :]
    y = np.expand_dims(y, axis=-1)
    y = np.expand_dims(y, axis=1)

    #Predicting
    pred = model.predict(x)

    #Visualizing predictions
    for i in range(len(pred)):     
        fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3.21, 4]})
        ax[0].imshow(y[i,0,:,:, 0], origin='lower')
        ax[0].set_title("Ground truth", fontsize=16)
        ax[0].axis('off')
        im=ax[1].imshow(pred[i,0,:,:,0], origin='lower') 
        ax[1].set_title("Prediction", fontsize=16)
        ax[1].axis('off')
        fig.tight_layout()
        fig.colorbar(im, shrink=0.71)
        plt.show()
