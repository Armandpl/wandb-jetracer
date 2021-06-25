import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np
from torchvision import transforms


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, train=True):
        super(XYDataset, self).__init__()
        self.directory = directory
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        self.refresh()

        # augment the training set only
        self.random_hflip = train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann["image_path"], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)

        x = 2.0 * (ann["x"] / width - 0.5)  # -1 left, +1 right
        y = 2.0 * (ann["y"] / height - 0.5)  # -1 top, +1 bottom

        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x

        return image, torch.Tensor([x, y])

    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split("_")
        x = items[0]
        y = items[1]
        return int(x), int(y)

    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, "*.jpg")):
            x, y = self._parse(image_path)
            self.annotations += [{"image_path": image_path, "x": x, "y": y}]

    def save_entry(self, category, image, x, y):
        if not os.path.exists(self.directory):
            subprocess.call(["mkdir", "-p", self.directory])

        filename = "%d_%d_%s.jpg" % (x, y, str(uuid.uuid1()))

        image_path = os.path.join(self.directory, filename)
        cv2.imwrite(image_path, image)
        self.refresh()


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    # image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
