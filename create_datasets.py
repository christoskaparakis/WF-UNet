import os
import h5py
import numpy as np
from tqdm import tqdm
import xarray as xr

directory = "dataset"
files_list = os.listdir(directory)
files_list.sort()
files_full_path = [os.path.join(directory, i) for i in files_list]
datasets = [xr.open_dataset(i) for i in files_full_path]
imgSize1 = 105
imgSize2 = 173
num_pixels = imgSize1 * imgSize2


def create_dataset(input_length, image_ahead, rain_amount_thresh):
    # Creating a dataset for training and testing.
    # create target file name
    filename = f"final_datasets/train_test_2016-2021_input-length_{input_length}_img-ahead_{image_ahead}_rain-threshhold_{int(rain_amount_thresh * 100)}.h5"

    # create and open h5 file
    with h5py.File(filename, "w", rdcc_nbytes=1024 ** 3) as f:

        ##############################################
        # train set

        train_set = f.create_group("train")
        train_timestamp_dataset = train_set.create_dataset(
            "timestamps",
            shape=(1, input_length + image_ahead, 1),
            maxshape=(None, input_length + image_ahead, 1),
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
            compression_opts=9,
        )
        train_tp_image_dataset = train_set.create_dataset(
            "tp_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        train_sp_image_dataset = train_set.create_dataset(
            "sp_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        train_u100_image_dataset = train_set.create_dataset(
            "u100_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        train_v100_image_dataset = train_set.create_dataset(
            "v100_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        train_t2m_image_dataset = train_set.create_dataset(
            "t2m_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )

        first = True
        for dataset in datasets:
            print(dataset)
            # training data
            dataset = dataset.sel(
                time=slice(None, "2020")
            )  # to exclude 2021 from the train set

            origin = [dataset.time, dataset.tp, dataset.sp, dataset.u100, dataset.v100, dataset.t2m]
            target = [
                train_timestamp_dataset,
                train_tp_image_dataset,
                train_sp_image_dataset,
                train_u100_image_dataset,
                train_v100_image_dataset,
                train_t2m_image_dataset
            ]

            timestamps, tp_images, sp_images, u100_images, v100_images, t2m_images = origin
            timestamp_dataset, tp_dataset, sp_dataset, u100_dataset, v100_dataset,t2m_dataset = target

            for i in tqdm(range(input_length + image_ahead, len(tp_images))):
                # If threshold of rain is bigger in the target image: add sequence to dataset
                if np.sum(tp_images[i] > 0) >= num_pixels * rain_amount_thresh:
                    tp_imgs = tp_images[i - (input_length + image_ahead) : i].values
                    timestamps_img = timestamps[
                        i - (input_length + image_ahead) : i
                    ].values
                    timestamps_img = timestamps_img.reshape((len(timestamps_img), 1))
                    # print(tp_imgs.shape)
                    # print(timestamps_img.shape)
                    # extend the dataset by 1 and add the entry
                    sp_imgs = sp_images[i - (input_length + image_ahead) : i].values
                    u100_imgs = u100_images[i - (input_length + image_ahead) : i].values
                    v100_imgs = v100_images[i - (input_length + image_ahead) : i].values
                    t2m_imgs = t2m_images[i - (input_length + image_ahead) : i].values

                    if first:
                        first = False
                    else:
                        timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                        tp_dataset.resize(tp_dataset.shape[0] + 1, axis=0)
                        sp_dataset.resize(sp_dataset.shape[0] + 1, axis=0)
                        u100_dataset.resize(u100_dataset.shape[0] + 1, axis=0)
                        v100_dataset.resize(v100_dataset.shape[0] + 1, axis=0)
                        t2m_dataset.resize(t2m_dataset.shape[0] + 1, axis=0)

                    timestamp_dataset[-1] = np.datetime_as_string(
                        timestamps_img, unit="h"
                    )
                    tp_dataset[-1] = tp_imgs
                    sp_dataset[-1] = sp_imgs
                    u100_dataset[-1] = u100_imgs
                    v100_dataset[-1] = v100_imgs
                    t2m_dataset[-1] = t2m_imgs
            print("Current size of training dataset:" + str(len(timestamp_dataset)))

        ###################################################
        # test set
        test_set = f.create_group("test")

        test_timestamp_dataset = test_set.create_dataset(
            "timestamps",
            shape=(1, input_length + image_ahead, 1),
            maxshape=(None, input_length + image_ahead, 1),
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
            compression_opts=9,
        )
        test_tp_image_dataset = test_set.create_dataset(
            "tp_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        test_sp_image_dataset = test_set.create_dataset(
            "sp_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        test_u100_image_dataset = test_set.create_dataset(
            "u100_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        test_v100_image_dataset = test_set.create_dataset(
            "v100_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        test_t2m_image_dataset = test_set.create_dataset(
            "t2m_images",
            shape=(1, input_length + image_ahead, imgSize1, imgSize2),
            maxshape=(None, input_length + image_ahead, imgSize1, imgSize2),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )

        # keep only 2021 as test
        test_dataset = datasets[-1].sel(time="2021")

        origin = [
            test_dataset.time,
            test_dataset.tp,
            test_dataset.sp,
            test_dataset.u100,
            test_dataset.v100,
            test_dataset.t2m
        ]
        target = [
            test_timestamp_dataset,
            test_tp_image_dataset,
            test_sp_image_dataset,
            test_u100_image_dataset,
            test_v100_image_dataset,
            test_t2m_image_dataset
        ]

        timestamps, tp_images, sp_images, u100_images, v100_images, t2m_images = origin
        timestamp_dataset, tp_dataset, sp_dataset, u100_dataset, v100_dataset, t2m_dataset = target
        first = True
        for i in tqdm(range(input_length + image_ahead, len(tp_images))):
            # If threshold of rain is bigger in the target image: add sequence to dataset
            if np.sum(tp_images[i] > 0) >= num_pixels * rain_amount_thresh:
                tp_imgs = tp_images[i - (input_length + image_ahead) : i]
                timestamps_img = timestamps[i - (input_length + image_ahead) : i].values
                timestamps_img = timestamps_img.reshape((len(timestamps_img), 1))
                # print(timestamps_img)
                #                     print(imgs.shape)
                #                     print(timestamps_img.shape)
                # extend the dataset by 1 and add the entry
                sp_imgs = sp_images[i - (input_length + image_ahead) : i]
                u100_imgs = u100_images[i - (input_length + image_ahead) : i]
                v100_imgs = v100_images[i - (input_length + image_ahead) : i]
                t2m_imgs = t2m_images[i - (input_length + image_ahead) : i]

                if first:
                    first = False
                else:
                    timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                    tp_dataset.resize(tp_dataset.shape[0] + 1, axis=0)
                    sp_dataset.resize(sp_dataset.shape[0] + 1, axis=0)
                    u100_dataset.resize(u100_dataset.shape[0] + 1, axis=0)
                    v100_dataset.resize(v100_dataset.shape[0] + 1, axis=0)
                    t2m_dataset.resize(t2m_dataset.shape[0] + 1, axis=0)

                timestamp_dataset[-1] = np.datetime_as_string(timestamps_img, unit="h")
                tp_dataset[-1] = tp_imgs
                sp_dataset[-1] = sp_imgs
                u100_dataset[-1] = u100_imgs
                v100_dataset[-1] = v100_imgs
                t2m_dataset[-1] = t2m_imgs
        print("Size of testing dataset:" + str(len(timestamp_dataset)))


if __name__ == "__main__":
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    # create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.5)
