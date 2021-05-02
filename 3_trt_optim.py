import os
import torch
import wandb
import torchvision
from torch2trt import torch2trt

if __name__ == "__main__":
    with wandb.init(project="racecar", job_type="trt-optimization", entity="wandb") as run:
        
        print("downloading non optimized model")
        artifact = run.use_artifact('model:latest')
        artifact_dir = artifact.download()


        print("creating model architecture")
        # fetching the model architecture from the producer run
        producer_run = artifact.logged_by()
        model = torchvision.models.__dict__[producer_run.config['architecture']](pretrained=False)
        
        device = torch.device('cuda')
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model = model.cuda().eval().half() 
        model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.pth')))

        # dummy input
        data = torch.zeros((1, 3, 224, 224)).cuda().half()

        print("optimizing model...")
        model_trt = torch2trt(model, [data], fp16_mode=True)

        # evaluate model
        # TODO

        # save the model for inference
        print("saving model and uploading it to wandb...")
        torch.save(model_trt.state_dict(), 'trt-model.pth')

        trt_artifact = wandb.Artifact('trt-model', type='model')
        trt_artifact.add_file('trt-model.pth')
        run.log_artifact(trt_artifact)

