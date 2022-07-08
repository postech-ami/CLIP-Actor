# CLIP-Actor 
### [Project Page](https://clip-actor.github.io) | [Paper](https://arxiv.org/abs/2206.04382)
This repository contains a pytorch implementation for the paper: [CLIP-Actor: Text-Driven Recommendation and Stylization for Animating Human Meshes](https://arxiv.org/abs/2206.04382). CLIP-Actor is a novel text-driven **motion recommendation** and **neural mesh stylization** system for human mesh animation.<br><br>

https://user-images.githubusercontent.com/55628873/173112861-93e07ac6-5303-44db-940c-68b75a947085.mp4

## Getting Started
This code was developed on Ubuntu 18.04 with Python 3.7, CUDA 10.2 and PyTorch 1.9.0. Later versions should work, but have not been tested.

### System Requirements
- Python 3.7
- CUDA 10.2
- Single GPU w/ minimum 24 GB RAM

### Environment Setup
Create and activate a virtual environment to work in, e.g. using Conda:
```
conda create -n clip_actor python=3.7
conda activate clip_actor
```

Install [PyTorch](https://pytorch.org/) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). For CUDA 10.2, this would look like:
```
conda install -c pytorch pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

You must also have [ffmpeg](https://ffmpeg.org/) installed on your system to save visualizations.

### Download Body Models and Datasets
To run CLIP-Actor, you need to download relevant body models and datasets. 

Check [DOWNLOAD.md](datasets/DOWNLOAD.md) for details.

## Running CLIP-Actor
Run below commands to generate whatever stylized 4D human avatar you want! 
```bash
python clip_actor.py --prompt "a scuba diver is scuba diving" --exp_name scuba_diving
python clip_actor.py --prompt "Freddie Mercury is dancing" --exp_name mercury_dancing
```
The outputs will be the final .mp4 video, stylized .obj files, colored render views, and screenshots during training.

## Citation
If you find our code or paper helps, please consider citing:
```
@inproceedings{youwang2022clipactor,
      title={CLIP-Actor: Text-Driven Recommendation and Stylization for Animating Human Meshes},
      author={Kim Youwang and Kim Ji-Yeon and Tae-Hyun Oh},
      year={2022},
      booktitle={ECCV}
}
```

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) 
grant funded by the Korea government(MSIT) (No.2022-00164860, Development of Human Digital Twin Technology Based on Dynamic Behavior Modeling and Human-Object-Space Interaction; and No.2021-0-02068, Artificial Intelligence Innovation Hub).

The implementation of CLIP-Actor is largely inspired and fine-tuned from the seminal prior work, [Text2Mesh](https://github.com/threedle/text2mesh) (Michael _et al._).
We thank the authors of Text2Mesh who made their code public. Also If you find these works helpful, please consider citing them as well.


