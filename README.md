# 🏁🏎️💨 = W&B ➕ NVIDIA Jetracer [![tests](https://github.com/Armandpl/wandb_jetracer/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Armandpl/wandb_jetracer/actions/workflows/ci.yml) [![flake8 Lint](https://github.com/Armandpl/wandb_jetracer/actions/workflows/lint.yml/badge.svg)](https://github.com/Armandpl/wandb_jetracer/actions/workflows/lint.yml) [![codecov](https://codecov.io/gh/Armandpl/wandb_jetracer/branch/master/graph/badge.svg?token=ZWFFBNQNWB)](https://codecov.io/gh/Armandpl/wandb_jetracer)

<p align="center"><img src="https://raw.githubusercontent.com/Armandpl/wandb-jetracer/master/assets/header.png"/></p>
<p align="center"><i>A picture of the car along with it's POV</i></p>

This project builds on top of the [NVIDIA Jetracer](https://github.com/NVIDIA-AI-IOT/jetracer) project to instrument it with [Weights&Biases](https://wandb.ai/site), making it easier to train, evaluate and refine models.  
It features a full pipeline to effortlessly `collect_data`, `label` it, `train`/`optimize` models and finally `drive` the car while monitoring how the model is doing.

<p align="center"><img src="https://raw.githubusercontent.com/Armandpl/wandb-jetracer/master/assets/artifacts.png" height="300"/></p>
<p align="center"><i><a href="https://docs.wandb.ai/guides/artifacts">Weights&Biases Artifacts'</a> graph showing the whole pipeline.</i></p>

## Pipeline overview
The repo is meant to be used by running (and modifying!) scripts under `src/wandb_jetracer/scripts`. These allow to train a model to detect the center of the racetrack which we can then feed to a control policy to drive the car. 
1. `collect_data.py` will take pictures using the car's camera and upload them to Weights&Biases. It should be ran while manually driving the car around.
2. `label.py` is a labelling utiliy. It will download the images from the previous step to a computer to annotate them with the relevant labels. The labels will then be added to the dataset stored on Weights&Biases servers. 
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Armandpl/wandb-jetracer/blob/master/src/wandb_jetracer/wandb_jetracer_training.ipynb)`wandb_jetracer_training.ipynb` is used to download the same dataset, train a model and upload it's weights to WandB.
4. `trt_optim` is meant to be ran on the car. It will convert the latest trained model to [TensorRT](https://developer.nvidia.com/tensorrt) for inference.
5. `drive.py` will take the optimized model and use it to drive the car. It will also log sensor data (IMU, Camera), system metrics ([jetson stats](https://github.com/rbonghi/jetson_stats), inference time) as well as the control signal to WandB. This helps with monitoring the model's perfomances in production.

## Setup and dependencies
These scripts are run on three different types of machine: the actual [embedded jetson nano computer](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) on the car, a machine used for labelling and a colab instance used for training.  
They all relie on different dependencies:
- Training dependencies are installed in the Colab notebook so you don't need to worry about those.
- Labelling dependencies can be installed in a conda env using `conda create -f labelling_env.yml`
- Dependencies for the car are slightly trickier to get right, you'll find instructions [here](). Feel free to open issues if you run into troubles!


### Disclaimer about default throttle values -> jetson_setup.md

## Testing

## Footnote
Feel free to open GitHub issues if you have any questions!

## Ressources
- Donkey car
- a1k0n repo?
- circuit launch?
- [wandb dashboard](https://wandb.ai/armandpl2/wandb-jetracer)  
- [notion doc](https://www.notion.so/wandbai/Self-Driving-RC-Car-25ec247621094e998a6ddbb7ee90ec93)  
- [original nvidia project](https://github.com/NVIDIA-AI-IOT/jetracer)  

todo:
- button to go to the colab notebook
- 
- link to the video
- describe the pipeline in details
