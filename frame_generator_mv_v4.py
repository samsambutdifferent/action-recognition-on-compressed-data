import random
import numpy as np
import tensorflow as tf

from mvextractor.videocap import VideoCap

MV_FRAME_IDENTIFIER = "P"

def format_motion_vectors(motion_vectors, height):
    """
    """
    if len(motion_vectors) > height:
        # remove extra motion vectors until value equals height
        motion_vectors = motion_vectors[:height]
    else:
        # Pad motion_vectors with an empty motion vector
        # all mvs are 10 long
        while len(motion_vectors) < height:
            motion_vectors = np.vstack((motion_vectors, np.zeros(10)))

    return motion_vectors


def mvs_from_video_file(video_path, n_frames, height):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """    
    mvs_lst = []

    cap = VideoCap()
    cap.open(str(video_path))

    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret = True

    while ret:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if ret and frame_type == MV_FRAME_IDENTIFIER:
            # # Pad width of Motion Vectors
            motion_vectors = format_motion_vectors(motion_vectors=motion_vectors, height=height)
            mvs_lst.append(motion_vectors)
                    
    cap.release()

    # Original array
    
    original_array = np.array(mvs_lst)

    # TODO why does this happen the shape can be (0,)
    # would mean it doesnt have any mv values..
    if original_array.shape == (0,):
        # Set original_array to new shape with zeros
        original_array = np.zeros((n_frames, height, 10))

    # Expanded array with zeros padding up to the max depth value n_frames
    expanded_array = np.zeros((n_frames, height, 10))
    expanded_array[:original_array.shape[0], :, :] = original_array

    mvs_lst = np.array(expanded_array)

    return mvs_lst


class FrameGenerator:
    def __init__(self, path, n_frames, height, index_df, training=False):
        """Returns a set of frames with their associated label.

        Args:
          path: Video file paths.
          n_frames: Number of frames.
          training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.height = height
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
            video_frames = mvs_from_video_file(
                video_path=path,
                n_frames=self.n_frames,
                height=self.height,
            )
            label = self.class_ids_for_name[name]  # Encode labels

            yield video_frames, label
