import random
import cv2
import numpy as np
import tensorflow as tf


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=2):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()  # FAILING HERE!!!
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, path, n_frames, index_df, frame_step, training=False, provide_file_paths=False):
        """Returns a set of frames with their associated label.

        Args:
          path: Video file paths.
          n_frames: Number of frames.
          training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.frame_step = frame_step
        self.training = training
        self.class_names = index_df["category"].unique()
        self.class_ids_for_name = dict(
            (name, idx) for idx, name in enumerate(self.class_names)
        )
        self.index_df = index_df
        self.provide_file_paths = provide_file_paths

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
                frame_step=self.frame_step,
            )
            label = self.class_ids_for_name[name]  # Encode labels

            if self.provide_file_paths:
                yield video_frames, label, str(path)
            else:
                yield video_frames, label