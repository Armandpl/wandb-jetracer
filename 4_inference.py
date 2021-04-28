import os
import cv2
import time
import torch
import wandb
import argparse
from utils import preprocess
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the optimized model on the car") 

    parser.add_argument("--framerate", type=int, default=10)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--debug_time", type=int, default=120, help="how long should it run for in seconds")
    parser.add_argument("--debug_freq", type=int, default=10, help="how many frame between each logged image")
    # TODO add policy as a param
    args = parser.parse_args()

    with wandb.init(project="wandb-jetracer", job_type="inference", config=args) as run:
        if run.config.debug:
            print("Debug mode enabled")
            frame_count = 0

        print("downloading latest optimized model")
        # artifact = run.use_artifact('trt-model:latest')
        # artifact_dir = artifact.download()

        print("loading state dict")
        model_trt = TRTModule()
        # model_trt.load_state_dict(torch.load(os.path.join(artifact_dir, 'trt-model.pth')))
        model_trt.load_state_dict(torch.load('trt-model.pth'))
        # model_trt.load_state_dict(torch.load('../jetracer/notebooks/road_following_model_trt.pth'))

        print("setting up car and camera")
        car = NvidiaRacecar()
        camera = CSICamera(width=224, height=224, capture_fps=run.config.framerate)

        car.throttle = -0.001
        STEERING_GAIN = -1
        print("all set")
        try:
            while True:
                image = camera.read()
                start = time.time()
                if run.config.debug:
                    unprocessed = image.copy()

                image = preprocess(image).half()
                output = model_trt(image).detach().cpu().numpy().flatten()
                x = float(output[0])
                y = float(output[1])
                car.steering = STEERING_GAIN*x # *(y+1)/2

                if run.config.debug: 
                    frame_count += 1
                    if frame_count % run.config.debug_freq == 0:
                        print("logging image")
                        x = int((x + 1)/2*224)
                        y = int((y + 1)/2*224)
                        cv2.circle(unprocessed, (x,y), 5, (0, 255, 0), 2)
                        unprocessed = cv2.cvtColor(unprocessed, cv2.COLOR_BGR2RGB)
                        wandb.log({"current_frame": wandb.Image(unprocessed)}) 

                    if frame_count == run.config.framerate*run.config.debug_time:
                        print("frame count: ", frame_count)
                        car.throttle=0.001
                        print("end debug")
                        break
                end = time.time()
                wandb.log({"system/inference_seconds": (end-start)})
                # print(x)
        except KeyboardInterrupt:
            pass
            
            
            

