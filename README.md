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

## Data evolution
Firstly, it is necessary to download:
+ bones.npy
+ jointAngleModel_v2.npy
+ staticPose.npy
+ template.npy  

from the [here](https://drive.google.com/drive/folders/1MUcR9oBNUpTAJ7YUWdyVLKCQW874FszI) and put it into the folder 
`${ROOT}/resources/constraints`.
To evolve the dataset run script evolve.py using the same parameter for the train.yaml, example

`python lib/evolution/evolve.py --cfg experiments/h36m/train.yaml`

It is necessary that data augmentation and occlusion parameter are set to False, other evolution parameters
are specified in lib/evolution/parameter.py. 

To use the evolved dataset one can make use of lib/evolution/evolution_dataset.py where the necessary parameters are
path to the normalized csv file and the path to unnormalized csv file, the class using the DataLoader returns joints_2d
and the joints_3d.


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
+ `bpeek` see output of jobs (`watch "bpeek | tail -n 30"` for continuous output)
+ `module load eth_proxy` allow network connection, necessary for downloading pre-trained models

## Recreating score

1. Clone this repo, and we will call the directory that you cloned as `${ROOT}`
2. Install dependencies using `pip`.
   
    `pip install -r requirements.txt`

### Training
3. To train the model run the following command:


    python scripts/train.py --cfg experiments/h36m/train.yaml
    
In order to train the model on Leonhard use:

    bsub -n 4 -W 6:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python scripts/train.py --cfg experiments/h36m/train.yaml

All parameters that produce the score are already set in `experiments/h36m/train.yaml`

To achieve the score we have trained the model for: TODO hours/stopped after TODO epochs

In order to resume training, one should change the `MODEL.RESUME` in `train.yaml` to the name of the checkpoint 
and run the model specifying the folder of the model. Model folder are named using timestamps. Possible call for 
resuming the model would be:

    python scripts/train.py --cfg experiments/h36m/train.yaml --model_time_str 2021-05-21-15-32-18

### Validation
4. To validate the model run the following command:


    python scripts/valid.py --cfg experiments/h36m/test.yaml

Or using Leonhard:

    bsub -n 4 -W 6:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python scripts/valid.py --cfg experiments/h36m/test.yaml

All necessary parameters are set in `test.yaml`