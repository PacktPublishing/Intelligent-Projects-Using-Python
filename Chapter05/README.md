# Chapter05: A Video Captioning Application

Video captioning, the art of translating a video to generate a meaningful summary of the
content, is a challenging task in the field of computer vision and machine learning.
Traditional methods of video captioning haven't produced many success stories. However,
with the recent boost in artificial intelligence aided by deep learning, video captioning has
recently gained a significant amount of attention. The power of convolutional neural
networks, along with recurrent neural networks, has made it possible to build end-to-end
enterprise-level video-captioning systems.

#### Goal 
- [x] To build a Video Captioning System
- [x] Learn the architectures of the Model and the technical knowhow

#### Dataset Link
[Data link for Videos] (http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)
[Data link for Captions] (https://github.com/jazzsaxmafia/video_to_sequence/files/387979/video_corpus.csv.zip)



#### Command to execute the video preprocessin Script VideoCaptioningPreProcessing.py

```bash

python VideoCaptioningPreProcessing.py process_main --video_dest '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/Video Captioning/data/' --feat_dir '/media/santanu/9eb9b6dc-b380-486e-b4fd-
c424a325b976/Video Captioning/features/' --temp_dest '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/Video Captioning/temp/' --img_dim 224 --channels 3 --batch_size=128 --frames_step 80

```

#### Command to execute train and evaluate the model on holdout dataset


```bash

python Video_seq2seq.py process_main --path_prj '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/Video Captioning/' --caption_file video_corpus.csv --feat_dir features --cnn_feat_dim 4096 --h_dim 512 --batch_size 32 --lstm_steps 80 --video_steps=80 --out_steps 20 --learning_rate 1e-4--epochs=100

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 






