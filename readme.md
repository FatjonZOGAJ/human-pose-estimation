# Human Pose Estimation: Terminators

To use pretrained models, download them in _lib/models/pretrained_data_.

To forward Tensorboard from Leonhard to Local start your ssh-session with e.g. _ssh leonhard -L 6008:localhost:6008_. Then you have to start Tensorboard on Leonhard with e.g. _tensorboard --logdir human_pose_estimation/output/h36m/pose3d_alexnet_reg/demo/logs/ --port 6008_ depending on how your repository is called. Afterwards, you can observe the training with the following link on your local machine _http://localhost:6008/_.