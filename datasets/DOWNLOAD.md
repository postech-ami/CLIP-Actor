# Download

### Getting Started

1. Create folders that store body models, datasets, and logs.
    ```bash
    export REPO_DIR=$PWD
    mkdir -p $REPO_DIR/body_models  # body models
    mkdir -p $REPO_DIR/datasets  # datasets
    mkdir -p $REPO_DIR/logs  # logs
    ```
2. Download body models. 

    Get yourself registered and download `models_smplx_v1_1.zip` from [SMPL-X](https://smpl-x.is.tue.mpg.de/), and place it at `${REPO_DIR}/body_models/smplx`.
 The data structure should follow the hierarchy below.
    ```bash
   ${REPO_DIR}
   |-- body_models
   |   |-- smplx
   |   |   |-- SMPLX_NEUTRAL.pkl
   |   |   |-- SMPLX_NEUTRAL.npz
   |   |   |-- ...
   |-- datasets
   |-- logs
   |-- README.md
   |-- ...
   ```

3. Download motion retrieval datasets.
    * Register and download all the available SMPL+H sequences from [AMASS](https://amass.is.tue.mpg.de/) (This would take long time).
    * Register and download `babel_v1-0_release.zip` from [BABEL](https://babel.is.tue.mpg.de/). Note that BABEL provides the _annotations_ for the AMASS dataset.

    Please place the datasets at `${REPO_DIR}/datasets`. 

    You also need to download our manually preprocessed BABEL raw labels,
   ```bash
   cd ${REPO_DIR}/datasets 
   wget https://www.dropbox.com/s/6p69k1tl1gk4lom/clip-actor_preprocess.zip
   unzip clip-actor_preprocess.zip
   ```
    Finally, you need to download `featp_2_fps.json` from [BABEL github](https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/data/featp_2_fps.json).   
  
    The data structure should follow the hierarchy below.
    ```bash
     ${REPO_DIR}
     |-- body_models
     |-- datasets
     |   |-- amass
     |   |   |-- ACCAD
     |   |   |-- BMLhandball
     |   |   |-- ...
     |   |   |-- ...
     |   |-- babel
     |   |   |-- train.json
     |   |   |-- val.json
     |   |   |-- ...
     |   |   |-- raw_label.npy 
     |   |   |-- encoded_raw_label.npy
     |   |   |-- featp_2_fps.json
     |-- logs
     |-- README.md
     |-- ...
     ```