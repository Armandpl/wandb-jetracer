import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np
import random
import albumentations as A

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, train=False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        keypoint_params=A.KeypointParams(format='xy'),
        )
        self.refresh()
        self.train = train

        self.augmentations = A.Compose([
            # A.RandomCrop(width=112, height=112, p=0.3),
            # A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.IAAPerspective (scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.3),
            A.MotionBlur(p=0.3),
            # A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.3)
            #A.OneOf([
            #    A.HueSaturationValue(p=0.5),
            #    A.RGBShift(p=0.7)
            #], p=1),
        ],
        keypoint_params=A.KeypointParams(format='xy'),
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height

        image = self.transform(image=np.array(image), keypoints=[])['image']

        if self.train:
            for i in range(10):
                transformed = self.augmentations(image=np.array(image), keypoints=[(ann['x'], ann['y'])])
                image = transformed['image']
                if len(transformed['keypoints']) > 0:
                    x, y = transformed['keypoints'][0]
                    break

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom

        return torch.tensor(image, dtype=torch.float), torch.Tensor([x, y])

    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return int(x), int(y)

    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]
        
    def save_entry(self, image, x, y):
        if not os.path.exists(self.directory):
            subprocess.call(['mkdir', '-p', self.directory])
            

        height, width, _ = image.shape

        x = int(x/width * 224)
        y = int(y/height * 224)
        filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
       
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)  
        image_path = os.path.join(self.directory, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        

class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap
