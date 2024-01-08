import random
import cv2
import numpy as np
import tensorflow as tf

from mvextractor.videocap import VideoCap

KEY_FRAME_IDENTIFIER = "I"

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def format_motion_vectors(mv, output_size):
    """
    """
    return mv


def frames_from_video_file(video_path, n_frames, output_size=(224, 224)):
    
    frames_lst = []

    cap = VideoCap()
    cap.open(str(video_path))

    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret = True

    while ret:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        # ret, frame_cv2 = src.read()

        if ret and frame_type == KEY_FRAME_IDENTIFIER:
            frame = format_frames(frame, output_size)
            frames_lst.append(frame)

    cap.release()

    if len(frames_lst) > n_frames:
        # remove extra frames until value equals n_frames
        frames_lst = frames_lst[:n_frames]
    else:
        # Pad frames_lst with zeros to match the length of n_frames
        while len(frames_lst) < n_frames:
            frames_lst.append(np.zeros_like(frames_lst[0]))

    frames_lst = np.array(frames_lst)[..., [2, 1, 0]]

    return frames_lst


class FrameGenerator:
    def __init__(self, path, n_frames, index_df, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = index_df["category"].unique()
        self.class_ids_for_name = dict(
            (name, idx) for idx, name in enumerate(self.class_names)
        )
        self.index_df = index_df

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob("*.avi"))
        names = [path_obj.name for path_obj in video_paths]
        classes = []
        for name in names:
            classes.append(
                self.index_df[self.index_df["name"] == name]["category"].to_string(
                    index=False
                )
            )

        return video_paths, classes

    def __call__(self):
        """calls when class is called"""
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(
                video_path=path,
                n_frames=self.n_frames,
                output_size=(224, 224),
            )
            label = self.class_ids_for_name[name]  # Encode labels

            yield video_frames, label
