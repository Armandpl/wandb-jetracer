# Jetson Nano setup
A lot of these dependencies are not pre-compiled for the Jetson Nano so expect this to take multiple hours.

## 1. Install Jetpack on your Jetson Nano
Jetpack is an image with Deep Learning related libs preinstalled.
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
Follow the instructions at [https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available)  
You may have issues installing torchvision in your virtualenv. Try this:  
```
cd torchvision
easy_install .  --user --install-dir ~/python-envs/rc/lib/python3.6/site-packages
```

## 4. Setup TensorRT
Follow the instructions at [https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/)

## 5. Setup the dependencies to interract with the IMU
Follow the instructions at [https://docs.donkeycar.com/parts/imu/](https://docs.donkeycar.com/parts/imu/).

## 6. Install Jetson Stats
Follow instructions at [https://github.com/rbonghi/jetson_stats](https://github.com/rbonghi/jetson_stats)

## 5. Link the preinstalled Opencv to your virtual env
Re-compiling cv2 takes a long time, so you can link the pre-compiled version to your virtualenv.  
`ln -s /usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-
ite-packages/cv2.cpython-36m-aarch64-linux-gnu.so`

## 6. Install jetracer, torch2trt and jetcam
1. Open a terminal and call the following to install the [JetCam](http://github.com/NVIDIA-AI-IOT/jetcam) Python package.

    ```bash
    cd $HOME
    git clone https://github.com/NVIDIA-AI-IOT/jetcam
    cd jetcam
    ~/python-envs/rc/bin/python setup.py install
    ```
    
2. Execute the following command to install the [torch2trt](http://github.com/NVIDIA-AI-IOT/torch2trt) Python package

    ```bash
    cd $HOME
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    ~/python-envs/rc/bin/python setup.py install
    ```
    
2. Execute the following in a terminal to install the [JetRacer](http://github.com/NVIDIA-AI-IOT/jetracer) package
 
     ```bash
     cd $HOME
     git clone https://github.com/NVIDIA-AI-IOT/jetracer
     cd jetracer
     ~/python-envs/rc/bin/python setup.py install
     ```

## 7. If you intend to use the object detections features from this repo:
Install the yolov5 pip package
`pip install yolov5`
You might miss libs to install/compile yolov5 dependencies:
```
sudo apt-get update sudo apt-get install -y build-essential libatlas-base-dev gfortran
```
Download yolov5s weights:
```
cd wandb-jetracer/src/scripts/
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```
