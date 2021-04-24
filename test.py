import torch
import os
import json
import random

from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms

# from model.vgg11 import affordance_model
# from model.vgg16 import affordance_model
# from model.mobilenet import affordance_model
from model.vgg11_size import affordance_model
from evaluation import *

DATA_DIR = './data/affordance'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def make_dataset(directory, labels_dic):
    images = []
    print(directory)
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            path = os.path.join(root, fname)
            try:
                item = (path, labels_dic[fname[:-4]]) # remove .png
                images.append(item)
            except:
                continue
    print(len(images))
    random.Random(4).shuffle(images)
    return images

def make_labels():
    def mean(scores):
        return np.mean([v for v in scores.values()])
    def mean2(scores):
        out = []
        for v in scores.values():
            out += v
        return np.mean(out)
    labels_path = os.path.join(DATA_DIR, 'labels.json')
    labels_dic = json.load(open(labels_path, 'rb'))
    out = {}
    for file_name, scores in labels_dic.items():
        pinch = mean(scores['pinch'])
        clench = mean(scores['clench'])
        poke = mean(scores['poke'])
        palm = mean(scores['palm'])
        familiarity = mean2(scores['familiarity'])
        size = scores['size']
        out[file_name] = [pinch, clench, poke, palm, familiarity, size]
    return out

class DatasetFolder():
    def __init__(self, directory, transform):
        samples = make_dataset(directory, make_labels())
        self.loader = default_loader
        self.transform = transform
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        target = torch.Tensor(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    model = torch.load('./ckpt/vgg11_size_0_fold_best.ckpt')
    model.eval()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Pad(512),
            transforms.CenterCrop(512),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    images = DatasetFolder('./data/affordance/test', data_transforms['test'])
    dataloader = torch.utils.data.DataLoader(images, batch_size=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            obj_size = labels[:, -1].to(device)
            labels = labels[:, :-2].to(device)

            pinch, clench, poke, palm = model(inputs, obj_size)

            pinch = pinch.cpu().numpy()
            clench = clench.cpu().numpy()
            poke = poke.cpu().numpy()
            palm = palm.cpu().numpy()

            if i == 0:
                all_preds = np.concatenate((pinch, clench, poke, palm), axis=-1)
                all_labels = labels.cpu().numpy()
            else:
                preds = np.concatenate((pinch, clench, poke, palm), axis=-1)
                all_preds = np.concatenate((all_preds, preds), axis=0)
                labels = labels.cpu().numpy()
                all_labels = np.concatenate((all_labels, labels), axis=0)

    mse, corr, acc = score_evaluation_from_np_batches(all_labels, all_preds)
    print(f'MSE: {mse}, CORRELATION: {corr}, ACCURACY: {acc}')