# action-recognition-on-compressed-data

Read/run Through the notebooks in order for setup and running

## Setup download and format data

0_setup.ipynb

1_fetch_right_left_up_down_from_GCS

2.1_Naive_Spatial_Compression

2.2_Naive_temporal_compression

All models in this section are based off the paper by Tran et al. (2018) with reference to the code on at (Tensorflow)[https://www.tensorflow.org/tutorials/video/video_classification]. The (mv-extractor)[https://github.com/LukasBommes/mv-extractor] library is used for key frame and motion vector extraction


## R(2+1)D Models


## Uncompressed data on R(2+1)D

3_Naive_temporal_compression

## Spatially Compressed data on R(2+1)D

3.1.1_2d_plus_1_rlud_1200_sp60
3.1.2_2d_plus_1_rlud_1200_sp30
3.1.3_2d_plus_1_rlud_1200_sp15
3.1.4_2d_plus_1_rlud_1200_sp7
3.1.5_2d_plus_1_rlud_1200_sp4


## Temporal Compressed data on R(2+1)D

3.2.1_2d_plus_1_rlud_1200_fpr2
3.2.1_2d_plus_1_rlud_1200_fpr3
3.2.1_2d_plus_1_rlud_1200_fpr5
3.2.1_2d_plus_1_rlud_1200_fpr7

## Key Frames and Motion Vectors

4.1_2d_plus_1_rlud_1200_kf
4.2_2d_plus_1_rlud_1200_mv
4.2_2d_plus_1_rlud_1200_mv
4.3_dual_input_rlud_1200_kf_mv_v2



REFERENCES:
- https://developer.qualcomm.com/software/ai-datasets/something-something
- https://github.com/LukasBommes/mv-extractor
- https://www.tensorflow.org/tutorials/video/video_classification
- Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y. and Paluri, M., 2018. A closer look at spatiotemporal convolutions for action recognition. Proceedings of the ieee conference on computer vision and pattern recognition. pp.6450–6459