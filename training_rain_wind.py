import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import *
from models import *
from generators import DataGeneratorPrecipitationData as DataGenerator

##---- Fix random seed ----##
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

##---- Reading data ----##

# Select dataset
filename = (
    "/workspace/persistent/ck/final_datasets/train_test_2016-2021_input-length_12_img-ahead_1_rain-threshhold_50.h5"
)


# Read dataset
try:
    f = h5py.File(filename, "r")
except:
    raise Exception(
        "\n\nNo data was found! Get and decompress the data as indicated first."
    )


# To numpy array
data_train = f["/train/tp_images"][:100,:,:96,:96]  # remove the slicing later
max_train = 0.024250908
data_train = data_train/max_train #normalize data


data_train1 = f["/train/u100_images"][:100,:,:96,:96]
data_train2 = f["/train/v100_images"][:100,:,:96,:96]
data_wind_speed = np.sqrt((data_train1**2)+data_train2**2)
data_wind_max = 42.895767
data_wind_speed = data_wind_speed/data_wind_max
del data_train1
del data_train2


# Pick validation set
p = 0.2

data_val = data_train[-int(len(data_train) * p) :]
data_train = data_train[: -int(len(data_train) * p)]

data_wind_speed_val = data_wind_speed[-int(len(data_wind_speed) * p) :]
data_wind_speed_train = data_wind_speed[: -int(len(data_wind_speed) * p)]




##---- Training model ----##

# Parameters network
lags = 12
lat = 96
long = 96
feats = 1 #if len(data_train.shape)==4 else data_train.shape[-1]
feats_out = 1
convFilters = 16
dropoutRate = 0.5
loss = "mse"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#
metric = 'mse'


# Parameters training
epochs = 200
batch_size = 2


# Denormalizing metric
custom_mse = MSE_denormalized(
    max_train, batch_size, reduction_sum=True, latitude=lat, longitude=long
) # value is the highest occurring value in the training set


# Instantiation
model = MIF_UNet(lags, lat, long, feats, feats_out, convFilters, dropoutRate)

model.compile(
    loss=custom_mse.mse_denormalized_per_image,
    optimizer=optimizer,
    metrics=[
        custom_mse.mse_denormalized_per_image,
        custom_mse.mse_denormalized_per_pixel,
    ],
)
model.summary()


# Checkpoint to save best model
filepath = "/workspace/persistent/ck/saved_models_precipitation/best_model.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)
early_stop= tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, min_delta=0.00001, verbose=1)
callbacks_list = [checkpoint,early_stop,reduce_lr]


# Instantiating generators
#for dual stream
training_generator = DataGenerator([data_train,data_wind_speed_train], batch_size, lags,multistream=True)
validation_generator = DataGenerator([data_val,data_wind_speed_val], batch_size, lags,multistream=True)


# Training
history = model.fit(
    training_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks_list,   
    use_multiprocessing=False,
)


# Showing training history
fig = plt.figure(figsize=(10, 7))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Training history")
plt.ylabel("Loss")
plt.xlabel("Epochs")
# plt.ylim(top=0.005, bottom=0)  # Limit
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig.subplots_adjust(right=0.80, top=0.88)
plt.grid(b=None)
plt.savefig("training_results_precipitation.png", dpi=300)
plt.show()

