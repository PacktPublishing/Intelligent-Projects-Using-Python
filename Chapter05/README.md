**Work in Progress**

*preprocess_vid_2_cnn_fc_feat.py*
It creates mutiple image frames from video and then the same is sampled at a frequency.(Select 1 image frame/40). These frame images are then passed through a pre trained VGG19 Network and the features out of the last fully connected layer is stored as output.

These CNN features per frame(x_t) would go to a LSTM of the Neural Translation Machine as features at each time step(t).

 
