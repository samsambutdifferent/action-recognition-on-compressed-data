import os
import subprocess

import pandas as pd
import cv2
import numpy as np

def validate_files_downloaded(data_target):
    indx_df = pd.read_csv(f"{data_target}/indx_df.csv")

    train_files = os.listdir(f"{data_target}/train/")
    test_files = os.listdir(f"{data_target}/test/")
    validation_files = os.listdir(f"{data_target}/validation/")

    train_indx = list(indx_df[indx_df["type"] == "train"]["name"])
    test_indx = list(indx_df[indx_df["type"] == "test"]["name"])
    validation_indx = list(indx_df[indx_df["type"] == "validation"]["name"])

    for t_idx in train_indx:
        if t_idx not in train_files:
            print(f"train file: {t_idx} not in downloads")
    for t_idx in test_indx:
        if t_idx not in test_files:
            print(f"test file: {t_idx} not in downloads")
    for t_idx in validation_indx:
        if t_idx not in validation_files:
            print(f"validation file: {t_idx} not in downloads")


def is_video_working(file_path):
    try:
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            # print("Error: Cannot open video file")
            return False

        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            # print("Error: Cannot read video frame")
            return False

        # Release the video capture object
        cap.release()

        # Video is working
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def check_video_files_are_working(test_data_folder):
    for file in os.listdir(test_data_folder):
        subprocess.call(
            f'ffmpeg -v error -i {test_data_folder + file} -f null -',
            shell=True
        )


def check_folder_is_working(folder_prefix, type):
    for file in os.listdir(folder_prefix + "/" + type):
        file_name = file.split('/')[-1]
        working = is_video_working(folder_prefix + "/" + type + "/" + file_name)
    
    check_video_files_are_working(folder_prefix + "/" + type + "/")


def get_video_lngth(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def get_video_lengths(folder_path):
    video_lengths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".avi"):
            video_path = os.path.join(folder_path, filename)

            length = get_video_lngth(video_path)
            if length == -1:
                continue

            video_lengths.append(length)
            

    return np.array(video_lengths)


if __name__=="__main__":
    validate_files_downloaded("data_trunc")
    validate_files_downloaded("data_trunc_120")
    validate_files_downloaded("data_trunc_temporal_0.5")