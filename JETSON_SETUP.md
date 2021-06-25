# Jetson Nano setup
A lot of these dependencies are not pre-compiled for the Jetson Nano so expect this to take multiple hours.

## 1. Install Jetpack on your Jetson Nano
Jetpack is an image with Deep Learning related libs preinstalled. already  
Follow the instructions at [https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

## 2. Setup a virtualenv
I recommend you use a virtual env to isolate this project dependencies from your system.  
```bash
sudo apt install python3-pip
sudo apt install -y python3-venv
python3 -m venv ~/python-envs/rc
source ~/python-envs/rc/bin/activate
```

## 3. Install torch and torchvision
Follow the instructions at

## 4. Setup TensorRT

## 5. Link the preinstalled Opencv to your virtual env
Re-compiling cv2 takes a long time, so you can link the pre-compiled version to your virtualenv.

## 6. If you intend to use the object detections features from this repo:
Install the yolov5 pip package
`pip install yolov5`
You might miss dependencies to install/compile yolov5 dependencies:
```
```
