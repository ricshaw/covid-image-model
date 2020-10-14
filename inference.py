import numpy as np
import os
import sys
import random
import argparse
from PIL import Image
from PIL.Image import fromarray

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from efficientnet_pytorch import EfficientNet


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seeded!')
seed_everything(42)

def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img

def dicom_image_loader(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img = np.uint8(255.0*img)
    img = Image.fromarray(img).convert("RGB")
    return img

def get_image(filename, transform, A_transform=None, centre_crop=True):
    image = default_image_loader(filename)
    if centre_crop:
        image = transforms.CenterCrop(min(image.size))(image)
    # A transform
    if A_transform is not None:
        image = np.array(image)
        image = A_transform(image=image)['image']
        image = Image.fromarray(image)
    # Transform
    image = transform(image)
    return image

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class Model(nn.Module):
    def __init__(self, encoder='efficientnet-b0'):
        super(Model, self).__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),}
        self.out_chns = 0
        self.net = EfficientNet.from_name(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_chns += n_channels_dict[encoder]
        self.fc = nn.Linear(self.out_chns, 1)
        self.dropouts = nn.ModuleList([nn.Dropout(config['dropout']) for _ in range(5)])

    def forward(self, image):
        x = self.net.extract_features(image)
        x = self.avg_pool(x)
        x = nn.Flatten()(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out

## Transforms
def get_transform(image_size, mode='test'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if mode=='val':
        transform = transforms.Compose([transforms.Resize(image_size, 3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])
    if mode=='test':
        transform = transforms.Compose([transforms.Resize(image_size, 3),
                                    transforms.Lambda(lambda image: torch.stack([
                                    transforms.ToTensor()(image),
                                    transforms.ToTensor()(image.rotate(90,resample=0)),
                                    transforms.ToTensor()(image.rotate(180,resample=0)),
                                    transforms.ToTensor()(image.rotate(270,resample=0)),
                                    transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM)),
                                    transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(90,resample=0)),
                                    transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(180,resample=0)),
                                    transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(270,resample=0)),
                                    ])),
                                    transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean, std)(image) for image in images]))
                                    ])
    return transform


## Input args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model directory', required=True)
parser.add_argument('--image', type=str, help='input image filename', required=True)
args = parser.parse_args()
print('model:', args.model)
print('image:', args.image)

## Config
MODEL_NAME = os.path.basename(os.path.abspath(args.model))
ENCODER = MODEL_NAME.split('-')[-3] + '-' + MODEL_NAME.split('-')[-2]
BATCH_SIZE = int(MODEL_NAME.split('-')[1][2::])
LR = float(MODEL_NAME.split('-')[2][2::])
DROPOUT = float(MODEL_NAME.split('-')[-4][2::])
IMAGE_SIZE = int(MODEL_NAME.split('-')[-1][2::])
config = dict(batch_size=BATCH_SIZE,
              lr=LR,
              dropout=DROPOUT,
              image_size=IMAGE_SIZE,
              encoder=ENCODER,)
print('Config:', config)

## Load input image
image = get_image(args.image,
                  transform=get_transform(config['image_size'], mode='test'),
                  A_transform=None,
                  centre_crop=True)
image = image.unsqueeze(0).cuda()
print('Input:', image.shape)

## Init model
model = Model(config['encoder']).cuda()
model = nn.DataParallel(model)

## Inference
FOLDS = 5
pred = 0.0
for fold in range(FOLDS):

    ## Load checkpoint
    MODEL_PATH = os.path.join(os.path.abspath(args.model), 'fold_%d.pth' % fold)
    print('Model path:', MODEL_PATH)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        sys.exit('No model found!')

    model.eval()
    with torch.no_grad():
        ## Test-time augmentation
        if len(image.size())==5:
            bs, n_crops, c, h, w = image.size()
            out = model(image.view(-1, c, h, w))
            out = out.view(bs, n_crops, -1).mean()
        else:
            out = model(image)
        out = torch.sigmoid(out)
        print('Fold %d:' % fold, out.item())
        pred += out.item()

pred /= FOLDS
print('Prediction:', pred)

## Clear up
del model
torch.cuda.empty_cache()
