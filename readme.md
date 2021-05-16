# Human Pose Estimation: Terminators

To use pretrained models, download them in _lib/models/pretrained_data_.

To forward Tensorboard from Leonhard to Local start your ssh-session with e.g. _ssh leonhard -L 6008:localhost:6008_. Then you have to start Tensorboard on Leonhard with e.g. _tensorboard --logdir human_pose_estimation/output/h36m/pose3d_alexnet_reg/demo/logs/ --port 6008_ depending on how your repository is called. Afterwards, you can observe the training with the following link on your local machine _http://localhost:6008/_.

## Data augmentation
Downloading the VOC dataset needed for occlusion.

`cd Terminators`

`wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`tar -xf VOCtrainval_11-May-2012.tar`

Set the DATASET.VOC parameter in the train.yaml to the location of VOC2012.
To use occlusion set the DATASET.OCCLUSION parameter to True in train.yaml

## Using CPU
In debug mode one can set in the train.yaml CPU parameter to True, default is False
and the rest of the code works t same way as before without using CPU.

