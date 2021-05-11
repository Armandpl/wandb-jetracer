import logging

# TODO: check if I can import without having torch installed?
import PIL.Image
import torch
import torchvision.transforms as transforms

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def setup_logging(config=None):
    if config is None:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG if config.debug else logging.INFO

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging_level
    )
