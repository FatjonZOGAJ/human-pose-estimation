# Human Pose Estimation: Terminators

To use pretrained models, download them in _lib/models/pretrained_data_.

To forward Tensorboard from Leonhard to Local start your ssh-session with e.g. _ssh leonhard -L 6008:localhost:6008_. Then you have to start Tensorboard on Leonhard with e.g. _tensorboard --logdir human_pose_estimation/output/h36m/pose3d_alexnet_reg/demo/logs/ --port 6008_ depending on how your repository is called. Afterwards, you can observe the training with the following link on your local machine _http://localhost:6008/_.

## Data augmentation
Downloading the VOC dataset needed for occlusion.

`cd Terminators`

`wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`tar --strip-component=1 -xf VOCtrainval_11-May-2012.tar`


Set the DATASET.VOC parameter in the train.yaml to the location of VOC2012.
To use occlusion set the DATASET.OCCLUSION parameter to True in train.yaml

## Using CPU
In debug mode one can set in the train.yaml CPU parameter to True, default is False
and the rest of the code works t same way as before without using CPU.

## Workflow
Running an experiment creates a directory with all the respective files using the date as its folder name.
This includes checkpoint (last model), model_best (model with best validation score), log file as well as the configuration train.yaml file.
The config file that gets saved includes the arguments that were used to overwrite it by passing in commandline arguments.

Important commandline arguments:
+ `cfg`: model backbone defined in the respective yaml file
+ `train_debug_mode`: DEBUG_MODE parameter used to train for 2 iterations with batch size 2 to see if the code finishes without errors
+ `model_time_str`: load previously trained model_best by passing the folder name (currently only works for validation)

Check `train.py.parse_args()` for more and add your own there.
Ensure that these overwrite the parameters in the config using `train.py.reset_config()`.
Make sure to consider default values so that when no argument is passed, the config file does not get overwritten.

`bash vim_last_log.sh` opens the last log file automatically to not have to enter it manually.

### Leonhard commands
+ `bsubs` add jobs
+ `bjobs` list jobs (`-l` for details)
+ `bkill 0` kill all jobs
+ `bpeek` see output of jobs